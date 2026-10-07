// Valida o time-stretch que preserva o tom (WSOLA) do audio_rate_control.h.
// Alimenta uma senoide em blocos (como o core entrega) e mede:
//   - a frequencia de saida (deve bater com a de entrada -> tom preservado);
//   - a duracao de saida (deve ser ~ratio x a de entrada -> tempo esticado);
//   - o nivel (sem buracos de amplitude).
#include "audio_rate_control.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kRate = 44100;
constexpr int kChunk = 735;   // ~ um quadro de audio a 60 fps

std::vector<short> make_sine(double hz, double seconds) {
    const int frames = (int)(seconds * kRate);
    std::vector<short> out((size_t)frames * 2);
    for (int n = 0; n < frames; ++n) {
        const short s = (short)std::lround(30000.0 * std::sin(2.0 * 3.14159265358979323846 * hz * n / kRate));
        out[(size_t)n * 2] = s;
        out[(size_t)n * 2 + 1] = s;
    }
    return out;
}

// Frequencia pela contagem de cruzamentos por zero (canal esquerdo).
double zero_cross_hz(const std::vector<short>& pcm) {
    const size_t frames = pcm.size() / 2;
    if (frames < 2) return 0.0;
    long crossings = 0;
    for (size_t n = 1; n < frames; ++n) {
        const int a = pcm[(n - 1) * 2];
        const int b = pcm[n * 2];
        if (a <= 0 && b > 0) ++crossings;
    }
    return (double)crossings * kRate / (double)frames;
}

// Menor RMS em janelas de 50 ms: detecta "buracos" (artefato de OLA).
double min_window_rms(const std::vector<short>& pcm, int window) {
    const size_t frames = pcm.size() / 2;
    double worst = 1e30;
    for (size_t start = 0; start + (size_t)window <= frames; start += (size_t)window) {
        double acc = 0.0;
        for (int n = 0; n < window; ++n) {
            const double v = pcm[(start + (size_t)n) * 2];
            acc += v * v;
        }
        worst = std::min(worst, std::sqrt(acc / window));
    }
    return worst;
}

void run(double ratio) {
    AudioTimeStretch ts;
    const std::vector<short> in = make_sine(440.0, 3.0);
    const int in_frames = (int)(in.size() / 2);

    std::vector<short> out;
    const auto t0 = std::chrono::steady_clock::now();
    int pos = 0;
    while (pos < in_frames) {
        const int n = std::min(kChunk, in_frames - pos);
        const int produced = ts.process(in.data() + (size_t)pos * 2, n, ratio);
        out.insert(out.end(), ts.out(), ts.out() + (size_t)produced * 2);
        pos += n;
    }
    const double proc_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    const int out_frames = (int)(out.size() / 2);

    const double hz = zero_cross_hz(out);
    const double dur_ratio = (double)out_frames / in_frames;
    const double min_rms = min_window_rms(out, kRate / 20);

    std::printf("ratio %.2f: entrada %.2fs -> saida %.2fs (dur %.3fx), "
                "freq %.1f Hz (era 440.0, erro %+.1f%%), RMS minimo %.0f (de 21213), "
                "processamento %.1f ms (%.1f%% de 1x tempo real)\n",
                ratio, in_frames / (double)kRate, out_frames / (double)kRate,
                dur_ratio, hz, (hz - 440.0) / 440.0 * 100.0, min_rms,
                proc_ms, proc_ms / (in_frames / (double)kRate) / 10.0);

    // Tom preservado: frequencia dentro de 2%.
    assert(std::fabs(hz - 440.0) / 440.0 < 0.02);
    // Tempo esticado pela razao pedida (tolerancia de meio quadro).
    assert(std::fabs(dur_ratio - ratio) < 0.02);
    // Sem buracos de amplitude.
    assert(min_rms > 15000.0);
}

// Exercita o caminho de integracao: AudioRateControl::process() + output().
// (O backend SDL le a saida por output(); um descuido ali manda um buffer
// vazio com tamanho > 0 para SDL_QueueAudio e derruba o processo.)
void run_via_rate_control() {
    setenv("RETRORUN_AUDIO_TIME_STRETCH", "1", 1);
    setenv("RETRORUN_AUDIO_RATE_CONTROL", "1", 1);
    AudioRateControl rc;
    rc.configure(kRate, 140);

    const std::vector<short> in = make_sine(440.0, 3.0);
    const int in_frames = (int)(in.size() / 2);
    std::vector<short> out;
    int produced_total = 0;
    int pos = 0;
    while (pos < in_frames) {
        const int n = std::min(kChunk, in_frames - pos);
        const int produced = rc.process(in.data() + (size_t)pos * 2, n, 0,
                                        std::chrono::steady_clock::now());
        const short* p = rc.output();
        assert(p != nullptr || produced == 0);
        out.insert(out.end(), p, p + (size_t)produced * 2);
        produced_total += produced;
        pos += n;
    }
    const double hz = zero_cross_hz(out);
    std::printf("via rate_control: %d frames de saida, freq %.1f Hz\n",
                produced_total, hz);
    assert(produced_total > 0);
    assert(std::fabs(hz - 440.0) / 440.0 < 0.02);
}

} // namespace

int main() {
    run(1.00);
    run(1.15);
    run(1.30);
    run(0.90);
    run_via_rate_control();
    std::printf("audio_time_stretch_test: OK\n");
    return 0;
}
