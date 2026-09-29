#include "benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <vector>

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

std::atomic<bool> g_benchmark_collecting{false};

namespace {

struct Counters {
    std::atomic<uint64_t> core_frames{0};
    std::atomic<uint64_t> video_callbacks{0};
    std::atomic<uint64_t> duplicated_frames{0};
    std::atomic<uint64_t> skipped_fixed{0};
    std::atomic<uint64_t> skipped_adaptive{0};
    std::atomic<uint64_t> skipped_fast_forward{0};
    std::atomic<uint64_t> skipped_presenter_queue{0};
    std::atomic<uint64_t> presented_frames{0};
    std::atomic<uint64_t> direct_frames{0};
    std::atomic<uint64_t> fallback_frames{0};
    std::atomic<uint64_t> missed_deadlines{0};
    std::atomic<uint64_t> audio_callbacks{0};
    std::atomic<uint64_t> audio_frames{0};
    std::atomic<uint64_t> presentation_rejections{0};
    std::atomic<uint64_t> backend_time_us{0};
};

struct CounterSnapshot {
    uint64_t core_frames = 0;
    uint64_t video_callbacks = 0;
    uint64_t duplicated_frames = 0;
    uint64_t skipped_fixed = 0;
    uint64_t skipped_adaptive = 0;
    uint64_t skipped_fast_forward = 0;
    uint64_t skipped_presenter_queue = 0;
    uint64_t presented_frames = 0;
    uint64_t direct_frames = 0;
    uint64_t fallback_frames = 0;
    uint64_t missed_deadlines = 0;
    uint64_t audio_callbacks = 0;
    uint64_t audio_frames = 0;
    uint64_t presentation_rejections = 0;
    uint64_t backend_time_us = 0;
};

struct State {
    BenchmarkOptions options;
    BenchmarkMetadata metadata;
    BenchmarkAudioDiagnostics audio;
    Counters counters;
    bool requested = false;
    bool started = false;
    bool completed = false;
    bool failed = false;
    bool confirm_input = false;
    double confirm_input_delay_seconds = 4.0;
    std::string failure;
    TimePoint warmup_started{};
    TimePoint measurement_started{};
    TimePoint measurement_deadline{};
    TimePoint measurement_ended{};
    // Janela rolante (RETRORUN_BENCHMARK_ROLLING): guarda so os ultimos
    // `rolling_seconds` de amostras. Cada frame registra o tempo e um snapshot
    // dos contadores cumulativos; no report a janela = (contador atual -
    // snapshot mais antigo da janela).
    bool rolling = false;
    double rolling_seconds = 0.0;
    std::deque<TimePoint> frame_times;
    std::deque<CounterSnapshot> frame_cums;
    std::mutex samples_mutex;
    std::deque<double> core_us;
    std::deque<double> video_us;
    std::deque<double> audio_us;
    std::deque<double> active_frame_us;
    std::deque<double> lateness_us;
};

State state;
std::atomic<uint64_t> active_generation{0};
std::atomic<uint64_t> next_generation{1};

thread_local struct FrameTiming {
    TimePoint frame_started{};
    TimePoint core_started{};
    TimePoint video_started{};
    TimePoint audio_started{};
    uint64_t core_us = 0;
    uint64_t video_us = 0;
    uint64_t audio_us = 0;
    bool frame_active = false;
    bool core_active = false;
    bool video_active = false;
    bool audio_active = false;
} timing;

uint64_t elapsed_us(TimePoint start, TimePoint end)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count());
}

void reset_counters()
{
    state.counters.core_frames = 0;
    state.counters.video_callbacks = 0;
    state.counters.duplicated_frames = 0;
    state.counters.skipped_fixed = 0;
    state.counters.skipped_adaptive = 0;
    state.counters.skipped_fast_forward = 0;
    state.counters.skipped_presenter_queue = 0;
    state.counters.presented_frames = 0;
    state.counters.direct_frames = 0;
    state.counters.fallback_frames = 0;
    state.counters.missed_deadlines = 0;
    state.counters.audio_callbacks = 0;
    state.counters.audio_frames = 0;
    state.counters.presentation_rejections = 0;
    state.counters.backend_time_us = 0;
    std::lock_guard<std::mutex> lock(state.samples_mutex);
    state.core_us.clear();
    state.video_us.clear();
    state.audio_us.clear();
    state.active_frame_us.clear();
    state.lateness_us.clear();
    state.frame_times.clear();
    state.frame_cums.clear();
}

double average(const std::deque<double>& values)
{
    if (values.empty()) return 0.0;
    double total = 0.0;
    for (double value : values) total += value;
    return total / static_cast<double>(values.size());
}

double percentile(std::deque<double> values, double p)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double index = p * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(index));
    const size_t upper = static_cast<size_t>(std::ceil(index));
    if (lower == upper) return values[lower];
    const double fraction = index - static_cast<double>(lower);
    return values[lower] + (values[upper] - values[lower]) * fraction;
}

std::string json_escape(const std::string& value)
{
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
        case '\"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20)
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned>(c) << std::dec;
            else
                out << static_cast<char>(c);
        }
    }
    return out.str();
}

const char* json_bool(bool value) { return value ? "true" : "false"; }

} // namespace

bool benchmark_requested() { return state.requested; }

bool benchmark_configure(const BenchmarkOptions& options, std::string* error)
{
    if (!std::isfinite(options.duration_seconds) || options.duration_seconds <= 0.0) {
        if (error) *error = "benchmark duration must be a finite positive number";
        return false;
    }
    if (!std::isfinite(options.warmup_seconds) || options.warmup_seconds < 0.0) {
        if (error) *error = "benchmark warm-up must be a finite non-negative number";
        return false;
    }
    state.options = options;
    state.requested = true;
    state.started = false;
    state.completed = false;
    state.failed = false;
    state.confirm_input = false;
    state.confirm_input_delay_seconds = 4.0;
    state.failure.clear();
    // RETRORUN_BENCHMARK_ROLLING=1: janela rolante -- guarda so os ultimos
    // `duration_seconds` de amostras (descarta o dado antigo) e reporta essa
    // janela ao fechar o jogo (nunca fecha sozinha).
    state.rolling = getenv("RETRORUN_BENCHMARK_ROLLING") != nullptr;
    state.rolling_seconds = state.rolling ? options.duration_seconds : 0.0;
    return true;
}

void benchmark_set_metadata(const BenchmarkMetadata& metadata) { state.metadata = metadata; }

void benchmark_update_av(double declared_fps, double sample_rate)
{
    state.metadata.declared_fps = declared_fps;
    state.metadata.sample_rate = sample_rate;
}

void benchmark_begin_warmup()
{
    if (!state.requested) return;
    g_benchmark_collecting.store(false, std::memory_order_relaxed);
    state.warmup_started = Clock::now();
    if (state.options.warmup_seconds == 0.0)
        benchmark_update_window();
}

void benchmark_set_confirm_input(bool enabled)
{
    state.confirm_input = enabled;
}

void benchmark_set_confirm_input_delay(double seconds)
{
    state.confirm_input_delay_seconds = seconds;
}

bool benchmark_confirm_input_enabled()
{
    return state.confirm_input;
}

double benchmark_confirm_input_delay()
{
    return state.confirm_input_delay_seconds;
}

bool benchmark_confirm_button_pressed(BenchmarkConfirmButton button)
{
    // These synthetic presses exist only during warm-up, after the save state
    // has been loaded and before any benchmark samples are collected. Keeping
    // them time-based makes every run start from the same interactive scene.
    if (!state.requested || !state.confirm_input || state.started ||
        state.warmup_started == TimePoint{})
        return false;

    const double elapsed = std::chrono::duration<double>(
        Clock::now() - state.warmup_started).count();
    const double delay = state.confirm_input_delay_seconds;
    if (button == BenchmarkConfirmButton::A)
        return elapsed >= delay && elapsed < delay + 0.25;
    return elapsed >= delay + 0.75 && elapsed < delay + 1.00;
}

bool benchmark_update_window()
{
    if (!state.requested || state.failed || state.completed) return false;
    const auto now = Clock::now();
    if (!state.started) {
        const auto warmup = std::chrono::duration<double>(state.options.warmup_seconds);
        if (now - state.warmup_started < warmup) return false;
        reset_counters();
        state.measurement_started = now;
        state.measurement_deadline = now + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(state.options.duration_seconds));
        state.started = true;
        active_generation.store(next_generation.fetch_add(1, std::memory_order_relaxed),
                                std::memory_order_release);
        g_benchmark_collecting.store(true, std::memory_order_release);
        return true;
    }
    if (state.rolling)
        return false;   // janela rolante: nunca "completa" sozinha; reporta ao fechar
    const bool frame_limit_reached =
        state.options.core_frames != 0 &&
        state.counters.core_frames.load(std::memory_order_relaxed) >=
            state.options.core_frames;
    if (frame_limit_reached || now >= state.measurement_deadline) {
        g_benchmark_collecting.store(false, std::memory_order_release);
        state.measurement_ended = now;
        state.completed = true;
    }
    return false;
}

bool benchmark_deadline_reached()
{
    if (!state.requested || state.failed) return false;
    benchmark_update_window();
    return state.completed;
}

bool benchmark_keep_running()
{
    static int enabled = -1;
    if (enabled == -1)
        enabled = getenv("RETRORUN_BENCHMARK_KEEP_RUNNING") != nullptr ? 1 : 0;
    return enabled == 1;
}

void benchmark_disable()
{
    state.requested = false;
}

void benchmark_abort(const std::string& reason)
{
    if (!state.requested) return;
    g_benchmark_collecting.store(false, std::memory_order_release);
    state.failed = true;
    state.failure = reason;
}

void benchmark_frame_begin()
{
    timing = {};
    timing.frame_started = Clock::now();
    timing.frame_active = true;
}

void benchmark_core_begin()
{
    timing.core_started = Clock::now();
    timing.core_active = true;
}

void benchmark_core_end()
{
    if (!timing.core_active) return;
    timing.core_us += elapsed_us(timing.core_started, Clock::now());
    timing.core_active = false;
    state.counters.core_frames.fetch_add(1, std::memory_order_relaxed);
}

void benchmark_frame_end(TimePoint deadline, bool paced_frame)
{
    if (!timing.frame_active) return;
    const auto now = Clock::now();
    const uint64_t active_us = elapsed_us(timing.frame_started, now);
    const uint64_t exclusive_core_us = timing.core_us > timing.video_us + timing.audio_us
        ? timing.core_us - timing.video_us - timing.audio_us : 0;
    const uint64_t overhead_us = active_us > timing.core_us ? active_us - timing.core_us : 0;
    double late_us = 0.0;
    if (paced_frame && now > deadline) {
        late_us = static_cast<double>(elapsed_us(deadline, now));
        state.counters.missed_deadlines.fetch_add(1, std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lock(state.samples_mutex);
        state.core_us.push_back(static_cast<double>(exclusive_core_us));
        state.video_us.push_back(static_cast<double>(timing.video_us));
        state.audio_us.push_back(static_cast<double>(timing.audio_us));
        state.active_frame_us.push_back(static_cast<double>(active_us));
        state.lateness_us.push_back(static_cast<double>(late_us));
        // Store frontend overhead as a sixth, derived vector would be wasteful;
        // it is reconstructed from active minus callback-inclusive core below.
        (void)overhead_us;
        if (state.rolling)
        {
            CounterSnapshot snap;
            snap.core_frames = state.counters.core_frames.load(std::memory_order_relaxed);
            snap.video_callbacks = state.counters.video_callbacks.load(std::memory_order_relaxed);
            snap.duplicated_frames = state.counters.duplicated_frames.load(std::memory_order_relaxed);
            snap.skipped_fixed = state.counters.skipped_fixed.load(std::memory_order_relaxed);
            snap.skipped_adaptive = state.counters.skipped_adaptive.load(std::memory_order_relaxed);
            snap.skipped_fast_forward = state.counters.skipped_fast_forward.load(std::memory_order_relaxed);
            snap.skipped_presenter_queue = state.counters.skipped_presenter_queue.load(std::memory_order_relaxed);
            snap.presented_frames = state.counters.presented_frames.load(std::memory_order_relaxed);
            snap.direct_frames = state.counters.direct_frames.load(std::memory_order_relaxed);
            snap.fallback_frames = state.counters.fallback_frames.load(std::memory_order_relaxed);
            snap.missed_deadlines = state.counters.missed_deadlines.load(std::memory_order_relaxed);
            snap.audio_callbacks = state.counters.audio_callbacks.load(std::memory_order_relaxed);
            snap.audio_frames = state.counters.audio_frames.load(std::memory_order_relaxed);
            snap.presentation_rejections = state.counters.presentation_rejections.load(std::memory_order_relaxed);
            snap.backend_time_us = state.counters.backend_time_us.load(std::memory_order_relaxed);
            state.frame_times.push_back(now);
            state.frame_cums.push_back(snap);
            if (state.rolling_seconds > 0.0)
            {
                const auto cutoff = now - std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(state.rolling_seconds));
                while (!state.frame_times.empty() && state.frame_times.front() < cutoff)
                {
                    state.frame_times.pop_front();
                    state.frame_cums.pop_front();
                    state.core_us.pop_front();
                    state.video_us.pop_front();
                    state.audio_us.pop_front();
                    state.active_frame_us.pop_front();
                    state.lateness_us.pop_front();
                }
            }
        }
    }
    timing.frame_active = false;
}

void benchmark_video_callback_begin()
{
    timing.video_started = Clock::now();
    timing.video_active = true;
    state.counters.video_callbacks.fetch_add(1, std::memory_order_relaxed);
}

void benchmark_video_callback_end()
{
    if (!timing.video_active) return;
    timing.video_us += elapsed_us(timing.video_started, Clock::now());
    timing.video_active = false;
}

void benchmark_audio_callback_begin()
{
    timing.audio_started = Clock::now();
    timing.audio_active = true;
    state.counters.audio_callbacks.fetch_add(1, std::memory_order_relaxed);
}

void benchmark_audio_callback_end(uint64_t frames)
{
    if (!timing.audio_active) return;
    timing.audio_us += elapsed_us(timing.audio_started, Clock::now());
    timing.audio_active = false;
    state.counters.audio_frames.fetch_add(frames, std::memory_order_relaxed);
}

void benchmark_video_duplicate()
{
    state.counters.duplicated_frames.fetch_add(1, std::memory_order_relaxed);
}

void benchmark_video_skipped(BenchmarkSkipReason reason)
{
    switch (reason) {
    case BenchmarkSkipReason::Fixed: ++state.counters.skipped_fixed; break;
    case BenchmarkSkipReason::Adaptive: ++state.counters.skipped_adaptive; break;
    case BenchmarkSkipReason::FastForward: ++state.counters.skipped_fast_forward; break;
    case BenchmarkSkipReason::PresenterQueue: ++state.counters.skipped_presenter_queue; break;
    }
}

uint64_t benchmark_capture_generation()
{
    if (!benchmark_collecting()) return 0;
    return active_generation.load(std::memory_order_acquire);
}

void benchmark_presentation_completed(BenchmarkPresentation kind, uint64_t backend_time_us,
                                      uint64_t generation)
{
    if (generation == 0 || generation != active_generation.load(std::memory_order_acquire))
        return;
    state.counters.presented_frames.fetch_add(1, std::memory_order_relaxed);
    if (kind == BenchmarkPresentation::Direct)
        state.counters.direct_frames.fetch_add(1, std::memory_order_relaxed);
    else
        state.counters.fallback_frames.fetch_add(1, std::memory_order_relaxed);
    state.counters.backend_time_us.fetch_add(backend_time_us, std::memory_order_relaxed);
}

void benchmark_presentation_rejected()
{
    state.counters.presentation_rejections.fetch_add(1, std::memory_order_relaxed);
    benchmark_video_skipped(BenchmarkSkipReason::PresenterQueue);
}

void benchmark_set_audio_diagnostics(const BenchmarkAudioDiagnostics& diagnostics)
{
    state.audio = diagnostics;
}

bool benchmark_finish_and_report()
{
    g_benchmark_collecting.store(false, std::memory_order_release);
    if (!state.requested) return true;
    if (state.rolling && state.started && !state.completed && !state.failed)
    {
        // Janela rolante: ao fechar o jogo reporta a janela atual (os ultimos
        // <= rolling_seconds), mesmo sem ter atingido o deadline.
        state.measurement_ended = Clock::now();
        state.completed = true;
    }
    if (state.failed || !state.started || !state.completed) {
        std::fprintf(stderr, "Benchmark failed: %s\n",
                     state.failure.empty() ? "measurement did not complete" : state.failure.c_str());
        active_generation.store(0, std::memory_order_release);
        return false;
    }

    std::deque<double> core, video, audio, active, lateness;
    CounterSnapshot base;
    TimePoint window_start{};
    {
        std::lock_guard<std::mutex> lock(state.samples_mutex);
        core = state.core_us;
        video = state.video_us;
        audio = state.audio_us;
        active = state.active_frame_us;
        lateness = state.lateness_us;
        if (state.rolling && !state.frame_cums.empty())
        {
            base = state.frame_cums.front();
            window_start = state.frame_times.front();
        }
    }
    auto win = [](uint64_t now, uint64_t base) { return now - base; };
    const uint64_t fixed = win(state.counters.skipped_fixed.load(), base.skipped_fixed);
    const uint64_t adaptive = win(state.counters.skipped_adaptive.load(), base.skipped_adaptive);
    const uint64_t fast = win(state.counters.skipped_fast_forward.load(), base.skipped_fast_forward);
    const uint64_t queue = win(state.counters.skipped_presenter_queue.load(), base.skipped_presenter_queue);
    const uint64_t skipped = fixed + adaptive + fast + queue;
    const uint64_t presented = win(state.counters.presented_frames.load(), base.presented_frames);
    const uint64_t direct = win(state.counters.direct_frames.load(), base.direct_frames);
    const uint64_t fallback = win(state.counters.fallback_frames.load(), base.fallback_frames);
    const uint64_t w_core_frames = win(state.counters.core_frames.load(), base.core_frames);
    const uint64_t w_video_callbacks = win(state.counters.video_callbacks.load(), base.video_callbacks);
    const uint64_t w_audio_callbacks = win(state.counters.audio_callbacks.load(), base.audio_callbacks);
    const uint64_t w_audio_frames = win(state.counters.audio_frames.load(), base.audio_frames);
    const uint64_t w_duplicated = win(state.counters.duplicated_frames.load(), base.duplicated_frames);
    const uint64_t w_rejections = win(state.counters.presentation_rejections.load(), base.presentation_rejections);
    const uint64_t w_missed = win(state.counters.missed_deadlines.load(), base.missed_deadlines);
    const double w_backend_us = static_cast<double>(win(state.counters.backend_time_us.load(), base.backend_time_us));
    if (presented != direct + fallback) {
        std::fprintf(stderr, "Benchmark failed: presentation accounting invariant violated\n");
        active_generation.store(0, std::memory_order_release);
        return false;
    }
    double overhead_total = 0.0;
    const size_t sample_count = std::min(active.size(), core.size());
    for (size_t i = 0; i < sample_count; ++i) {
        const double used = core[i] + (i < video.size() ? video[i] : 0.0) +
                            (i < audio.size() ? audio[i] : 0.0);
        overhead_total += std::max(0.0, active[i] - used);
    }
    const double overhead_average = sample_count ? overhead_total / sample_count : 0.0;
    const double requested_duration = state.options.duration_seconds;
    const uint64_t requested_core_frames = state.options.core_frames;
    const double duration = (state.rolling && window_start != TimePoint{})
        ? std::chrono::duration<double>(state.measurement_ended - window_start).count()
        : std::chrono::duration_cast<std::chrono::duration<double>>(
              state.measurement_ended - state.measurement_started).count();
    const double max_lateness = lateness.empty() ? 0.0
        : *std::max_element(lateness.begin(), lateness.end());

    std::printf("\nRetroRun benchmark (%.3f seconds)\n", duration);
    std::printf("Core frames:              %llu\n", (unsigned long long)w_core_frames);
    std::printf("Video callbacks:          %llu\n", (unsigned long long)w_video_callbacks);
    std::printf("Presented frames:         %llu\n", (unsigned long long)presented);
    std::printf("Skipped frames:           %llu\n", (unsigned long long)skipped);
    std::printf("Duplicated frames:        %llu\n", (unsigned long long)w_duplicated);
    std::printf("Average core time:        %.3f ms\n", average(core) / 1000.0);
    std::printf("Average video time:       %.3f ms\n", average(video) / 1000.0);
    std::printf("Average audio time:       %.3f ms\n", average(audio) / 1000.0);
    std::printf("Average frontend overhead: %.3f ms\n", overhead_average / 1000.0);
    std::printf("Missed deadlines:         %llu\n", (unsigned long long)w_missed);
    std::printf("Audio underruns:          %llu\n", (unsigned long long)state.audio.buffer_underruns);
    std::printf("Direct scanout frames:    %llu\n", (unsigned long long)direct);
    std::printf("Fallback frames:          %llu\n", (unsigned long long)fallback);

    std::ostringstream json;
    json << std::fixed << std::setprecision(3)
         << "{\"release\":\"" << json_escape(state.metadata.release)
         << "\",\"device\":\"" << json_escape(state.metadata.device)
         << "\",\"backend\":\"" << json_escape(state.metadata.backend)
         << "\",\"audio_backend\":\"" << json_escape(state.metadata.audio_backend)
         << "\",\"renderer\":\"" << json_escape(state.metadata.renderer)
         << "\",\"core_name\":\"" << json_escape(state.metadata.core_name)
         << "\",\"core_version\":\"" << json_escape(state.metadata.core_version)
         << "\",\"declared_fps\":" << state.metadata.declared_fps
         << ",\"sample_rate\":" << state.metadata.sample_rate
         << ",\"duration_seconds\":" << duration
         << ",\"requested_duration_seconds\":" << requested_duration
         << ",\"requested_core_frames\":" << requested_core_frames
         << ",\"settings\":{\"declared_fps_pacing\":" << json_bool(state.metadata.declared_fps_pacing)
         << ",\"threaded_audio\":" << json_bool(state.metadata.threaded_audio)
         << ",\"stable_audio_buffer\":" << json_bool(state.metadata.stable_audio_buffer)
         << ",\"audio_buffer\":" << state.metadata.audio_buffer
         << ",\"sdl_audio_stretch_percent\":" << state.metadata.sdl_audio_stretch_percent
         << ",\"sdl_audio_stretch_low_ms\":" << state.metadata.sdl_audio_stretch_low_ms
         << ",\"go2_audio_stretch_percent\":" << state.metadata.go2_audio_stretch_percent
         << ",\"go2_audio_stretch_low_ms\":" << state.metadata.go2_audio_stretch_low_ms
         << ",\"go2_audio_prebuffer_ms\":" << state.metadata.go2_audio_prebuffer_ms
         << ",\"threaded_video\":" << json_bool(state.metadata.threaded_video)
         << ",\"direct_scanout\":" << json_bool(state.metadata.direct_scanout)
         << ",\"overlays\":" << json_bool(state.metadata.overlays)
         << ",\"vsync_requested\":" << json_bool(state.metadata.vsync_requested)
         << ",\"vsync_applied\":" << json_bool(state.metadata.vsync_applied)
         << ",\"fixed_frameskip\":" << state.metadata.fixed_frameskip
         << ",\"adaptive_frameskip\":" << json_bool(state.metadata.adaptive_frameskip)
         << ",\"confirm_input\":" << json_bool(state.metadata.confirm_input)
         << ",\"confirm_input_delay_seconds\":" << state.metadata.confirm_input_delay_seconds << "}"
         << ",\"counters\":{\"core_frames\":" << w_core_frames
         << ",\"video_callbacks\":" << w_video_callbacks
         << ",\"audio_callbacks\":" << w_audio_callbacks
         << ",\"audio_frames\":" << w_audio_frames
         << ",\"presented_frames\":" << presented
         << ",\"direct_scanout_frames\":" << direct
         << ",\"fallback_frames\":" << fallback
         << ",\"duplicated_frames\":" << w_duplicated
         << ",\"skipped_frames\":" << skipped
         << ",\"skipped_fixed\":" << fixed
         << ",\"skipped_adaptive\":" << adaptive
         << ",\"skipped_fast_forward\":" << fast
         << ",\"skipped_presenter_queue\":" << queue
         << ",\"presentation_rejections\":" << w_rejections
         << ",\"missed_deadlines\":" << w_missed << "}"
         << ",\"timing_ms\":{\"core_average\":" << average(core) / 1000.0
         << ",\"video_average\":" << average(video) / 1000.0
         << ",\"audio_average\":" << average(audio) / 1000.0
         << ",\"frontend_overhead_average\":" << overhead_average / 1000.0
         << ",\"backend_total\":" << w_backend_us / 1000.0
         << ",\"core_p50\":" << percentile(core, .50) / 1000.0
         << ",\"core_p95\":" << percentile(core, .95) / 1000.0
         << ",\"core_p99\":" << percentile(core, .99) / 1000.0
         << ",\"video_p50\":" << percentile(video, .50) / 1000.0
         << ",\"video_p95\":" << percentile(video, .95) / 1000.0
         << ",\"video_p99\":" << percentile(video, .99) / 1000.0
         << ",\"audio_p50\":" << percentile(audio, .50) / 1000.0
         << ",\"audio_p95\":" << percentile(audio, .95) / 1000.0
         << ",\"audio_p99\":" << percentile(audio, .99) / 1000.0
         << ",\"active_frame_p50\":" << percentile(active, .50) / 1000.0
         << ",\"active_frame_p95\":" << percentile(active, .95) / 1000.0
         << ",\"active_frame_p99\":" << percentile(active, .99) / 1000.0
         << ",\"lateness_p50\":" << percentile(lateness, .50) / 1000.0
         << ",\"lateness_p95\":" << percentile(lateness, .95) / 1000.0
         << ",\"lateness_p99\":" << percentile(lateness, .99) / 1000.0
         << ",\"lateness_max\":" << max_lateness / 1000.0 << "}"
         << ",\"audio\":{\"queued_frames\":" << state.audio.queued_frames
         << ",\"in_flight_frames\":" << state.audio.in_flight_frames
         << ",\"buffer_underruns\":" << state.audio.buffer_underruns
         << ",\"buffer_overruns\":" << state.audio.buffer_overruns
         << ",\"frames_dropped\":" << state.audio.frames_dropped
         << ",\"intentional_muted_frames\":" << state.audio.intentional_muted_frames
         << ",\"flush_discarded_frames\":" << state.audio.flush_discarded_frames
         << ",\"backpressure_events\":" << state.audio.backpressure_events
         << ",\"max_queue_depth\":" << state.audio.max_queue_depth
         << ",\"min_queue_depth\":" << state.audio.min_queue_depth
         << ",\"queue_depth_samples\":" << state.audio.queue_depth_samples
         << ",\"queue_depth_total_frames\":" << state.audio.queue_depth_total_frames
         << ",\"queue_empty_observations\":" << state.audio.queue_empty_observations
         << ",\"queue_low_observations\":" << state.audio.queue_low_observations
         << ",\"adaptive_stretch_frames\":" << state.audio.adaptive_stretch_frames
         << ",\"callback_max_duration_us\":" << state.audio.callback_max_duration_us
         << ",\"producer_gap_max_us\":" << state.audio.producer_gap_max_us
         << ",\"producer_late_max_us\":" << state.audio.producer_late_max_us
         << ",\"producer_late_over_1ms\":" << state.audio.producer_late_over_1ms
         << ",\"producer_late_over_5ms\":" << state.audio.producer_late_over_5ms
         << ",\"producer_late_over_10ms\":" << state.audio.producer_late_over_10ms
         << ",\"producer_late_over_20ms\":" << state.audio.producer_late_over_20ms
         << ",\"backend_submit_time_us\":" << state.audio.backend_submit_time_us
         << ",\"backend_submit_max_duration_us\":" << state.audio.backend_submit_max_duration_us
         << ",\"backend_submit_over_1ms\":" << state.audio.backend_submit_over_1ms
         << ",\"backend_submit_over_5ms\":" << state.audio.backend_submit_over_5ms
         << ",\"backend_submit_over_10ms\":" << state.audio.backend_submit_over_10ms
         << ",\"backend_submit_over_20ms\":" << state.audio.backend_submit_over_20ms << "}}";

    const std::string line = "BENCHMARK_JSON:" + json.str();
    std::printf("%s\n", line.c_str());
    if (!state.options.json_path.empty()) {
        std::ofstream output(state.options.json_path, std::ios::out | std::ios::trunc);
        if (!output) {
            std::fprintf(stderr, "Benchmark failed: cannot write JSON output '%s'\n",
                         state.options.json_path.c_str());
            active_generation.store(0, std::memory_order_release);
            return false;
        }
        output << json.str() << '\n';
    }
    active_generation.store(0, std::memory_order_release);
    return true;
}
