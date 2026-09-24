#pragma once

// Audio rate control for queue-based backends (SDL_QueueAudio).
//
// The core produces audio at the emulated rate. When the game runs below full
// speed the device queue drains and the backend used to insert a whole period
// of silence (~46 ms), heard as a crackle; small frame rate deviations did the
// same. Instead, every chunk is resampled before it is queued:
//
//   ratio = 1 / measured game speed      (long slow sections: the music just
//                                          plays slower, like a tape)
//         * (1 + gain * queue error)     (small deviations: +-1% around half
//                                          the target queue level)
//
// The game speed is the frames the core produced per second of real time,
// smoothed over ~0.7 s, not counting the time the backend itself blocked the
// producer (backpressure), so a full queue never looks like a slow game.
// The resampler is Catmull-Rom with state carried across chunks (no seams).
//
// Environment: RETRORUN_AUDIO_RATE_CONTROL=0 disables it;
// RETRORUN_AUDIO_MIN_SPEED_PERCENT (default 50) is the slowest playback;
// RETRORUN_AUDIO_DRC_PERMILLE (default 10) is the queue correction.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

class AudioRateControl {
public:
    void configure(int frequency, int target_queue_ms) {
        frequency_ = frequency;
        enabled_ = env_int("RETRORUN_AUDIO_RATE_CONTROL", 1, 0, 1) != 0;
        max_ratio_ = 100.0 / env_int("RETRORUN_AUDIO_MIN_SPEED_PERCENT", 50, 10, 100);
        gain_ = env_int("RETRORUN_AUDIO_DRC_PERMILLE", 10, 0, 50) / 1000.0;
        target_frames_ = static_cast<double>(frequency) *
            std::max(20, target_queue_ms / 2) / 1000.0;
        reset();
        std::fprintf(stderr,
            "RetroRun audio rate control: %s, min_speed=%.0f%%, drc=%.1f%%, target=%.0f frames\n",
            enabled_ ? "on" : "off", 100.0 / max_ratio_, gain_ * 100.0, target_frames_);
    }

    bool enabled() const { return enabled_; }

    // queue flushed: start over (history and speed)
    void reset() {
        history_valid_ = false;
        pos_ = 0.0;
        break_measurement();
    }

    // pause/resume: a pause is not a slow game
    void break_measurement() {
        speed_frames_ = 0.0;
        speed_seconds_ = 0.0;
        last_end_ = {};
    }

    // Resamples one chunk; the result is output()/output_frames.
    // call_start must be taken before any backpressure wait in the submit.
    int process(const short* data, int frames, uint64_t queued_frames,
                std::chrono::steady_clock::time_point call_start) {
        if (last_end_.time_since_epoch().count() != 0) {
            double dt = std::chrono::duration<double>(call_start - last_end_).count();
            dt = std::clamp(dt, 0.0, 0.25);
            const double decay = std::exp(-dt / kSpeedTauSeconds);
            speed_frames_ = speed_frames_ * decay + frames;
            speed_seconds_ = speed_seconds_ * decay + dt;
        }
        double base = 1.0;
        if (speed_seconds_ > 0.15 && frequency_ > 0) {
            const double speed = speed_frames_ / (speed_seconds_ * frequency_);
            base = 1.0 / std::clamp(speed, 1.0 / max_ratio_, 1.0);
        }
        const double error = target_frames_ > 0.0
            ? std::clamp((target_frames_ - static_cast<double>(queued_frames)) / target_frames_,
                         -1.0, 1.0)
            : 0.0;
        const double ratio = std::clamp(base * (1.0 + gain_ * error), 1.0 - gain_, max_ratio_);
        ratio_min = std::min(ratio_min, ratio);
        ratio_max = std::max(ratio_max, ratio);
        ratio_sum += ratio;
        ratio_count++;
        // tempo de audio por faixa de taxa (frames de entrada)
        const int bucket = ratio < 1.0 ? 0 : ratio < 1.02 ? 1 : ratio < 1.1 ? 2 :
                           ratio < 1.3 ? 3 : ratio < 1.6 ? 4 : 5;
        ratio_hist[bucket] += static_cast<uint64_t>(frames);
        if (ratio > 1.3 && first_slow_chunk == 0)
            first_slow_chunk = ratio_count;

        // input = 3 history frames + this chunk; pos is relative to frame 0,
        // interpolation between pos+1 and pos+2
        if (!history_valid_) {
            for (int i = 0; i < 3; ++i) {
                history_[i * 2] = data[0];
                history_[i * 2 + 1] = data[1];
            }
            pos_ = 0.0;
            history_valid_ = true;
        }
        const int total = frames + 3;
        input_.resize(static_cast<size_t>(total) * 2);
        std::copy(history_, history_ + 6, input_.begin());
        std::copy(data, data + static_cast<size_t>(frames) * 2, input_.begin() + 6);
        const short* in = input_.data();
        output_.resize(static_cast<size_t>(frames * max_ratio_ + 8) * 2);
        const int max_out = static_cast<int>(output_.size() / 2);
        const double step = 1.0 / ratio;
        double pos = pos_;
        int out = 0;
        while (pos < static_cast<double>(total - 3) && out < max_out) {
            const int i = static_cast<int>(pos);
            const float t = static_cast<float>(pos - i);
            for (int c = 0; c < 2; ++c) {
                const float v = catmull_rom(in[i * 2 + c], in[(i + 1) * 2 + c],
                                            in[(i + 2) * 2 + c], in[(i + 3) * 2 + c], t);
                output_[static_cast<size_t>(out) * 2 + c] =
                    static_cast<short>(std::clamp(std::lround(v), -32768L, 32767L));
            }
            ++out;
            pos += step;
        }
        pos_ = pos - static_cast<double>(frames);
        std::copy(in + static_cast<size_t>(frames) * 2, in + static_cast<size_t>(total) * 2,
                  history_);
        if (out > frames)
            frames_added += static_cast<uint64_t>(out - frames);
        output_frames = out;
        return out;
    }

    const short* output() const { return output_.data(); }

    // after the chunk was queued
    void submitted() { last_end_ = std::chrono::steady_clock::now(); }

    void report() const {
        if (!enabled_ || ratio_count == 0)
            return;
        std::fprintf(stderr,
            "RetroRun audio rate control: ratio avg %.3f min %.3f max %.3f, frames added %llu\n",
            ratio_sum / ratio_count, ratio_min, ratio_max,
            static_cast<unsigned long long>(frames_added));
        uint64_t total = 0;
        for (uint64_t v : ratio_hist)
            total += v;
        if (total == 0)
            return;
        std::fprintf(stderr,
            "RetroRun audio rate control: time at ratio <1.0 %.1f%%, 1.0-1.02 %.1f%%, 1.02-1.1 %.1f%%, "
            "1.1-1.3 %.1f%%, 1.3-1.6 %.1f%%, >=1.6 %.1f%% (first >1.3 at chunk %llu of %llu)\n",
            100.0 * ratio_hist[0] / total, 100.0 * ratio_hist[1] / total, 100.0 * ratio_hist[2] / total,
            100.0 * ratio_hist[3] / total, 100.0 * ratio_hist[4] / total, 100.0 * ratio_hist[5] / total,
            static_cast<unsigned long long>(first_slow_chunk), static_cast<unsigned long long>(ratio_count));
    }

    int output_frames = 0;
    uint64_t frames_added = 0;
    double ratio_min = std::numeric_limits<double>::max();
    double ratio_max = 0.0;
    double ratio_sum = 0.0;
    uint64_t ratio_count = 0;
    uint64_t ratio_hist[6] = {};
    uint64_t first_slow_chunk = 0;

private:
    static constexpr double kSpeedTauSeconds = 0.7;

    static int env_int(const char* name, int fallback, int lo, int hi) {
        const char* value = std::getenv(name);
        if (!value || !*value)
            return fallback;
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0')
            return fallback;
        return static_cast<int>(std::clamp<long>(parsed, lo, hi));
    }

    static float catmull_rom(float p0, float p1, float p2, float p3, float t) {
        return p1 + 0.5f * t * (p2 - p0 + t * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 +
                                             t * (3.0f * (p1 - p2) + p3 - p0)));
    }

    int frequency_ = 0;
    bool enabled_ = true;
    double max_ratio_ = 2.0;
    double gain_ = 0.01;
    double target_frames_ = 0.0;
    double speed_frames_ = 0.0;
    double speed_seconds_ = 0.0;
    std::chrono::steady_clock::time_point last_end_{};
    double pos_ = 0.0;
    short history_[6] = {};
    bool history_valid_ = false;
    std::vector<short> input_;
    std::vector<short> output_;
};
