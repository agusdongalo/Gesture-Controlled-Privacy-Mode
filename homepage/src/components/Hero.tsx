import { ArrowRight, Camera, Play } from 'lucide-react';

interface HeroProps {
  onOpenDemo: () => void;
}

export default function Hero({ onOpenDemo }: HeroProps) {
  return (
    <section className="relative min-h-screen flex items-center overflow-hidden pt-20">
      {/* Background grid */}
      <div className="absolute inset-0 opacity-[0.03]">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage:
              'linear-gradient(rgba(0,229,255,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(0,229,255,0.3) 1px, transparent 1px)',
            backgroundSize: '60px 60px',
          }}
        />
      </div>

      {/* Radial glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-cyan-500/5 rounded-full blur-[120px]" />
      <div className="absolute bottom-0 right-0 w-[500px] h-[500px] bg-violet-500/5 rounded-full blur-[100px]" />

      {/* Floating particles */}
      <div className="absolute top-20 left-[10%] w-2 h-2 bg-cyan-400/30 rounded-full animate-float" />
      <div className="absolute top-40 right-[15%] w-1.5 h-1.5 bg-violet-400/30 rounded-full animate-float" style={{ animationDelay: '2s' }} />
      <div className="absolute bottom-32 left-[20%] w-1 h-1 bg-cyan-400/20 rounded-full animate-float" style={{ animationDelay: '4s' }} />
      <div className="absolute top-60 left-[60%] w-2.5 h-2.5 bg-cyan-400/15 rounded-full animate-float" style={{ animationDelay: '1s' }} />

      <div className="relative mx-auto max-w-7xl px-6 w-full">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left — Text */}
          <div className="space-y-8 animate-fade-in-up">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/20">
              <span className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
              <span className="text-xs font-medium text-cyan-400 tracking-wide uppercase">
                Open Source · Python · MediaPipe
              </span>
            </div>

            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold leading-[1.08] tracking-tight">
              <span className="text-white">Privacy at</span>
              <br />
              <span className="gradient-text">Your Fingertips</span>
            </h1>

            <p className="text-lg text-slate-400 leading-relaxed max-w-lg">
              A real-time webcam privacy application that uses{' '}
              <span className="text-cyan-400 font-medium">hand gestures</span> to instantly toggle
              background blur, face pixelation, custom backgrounds, and sign-language modes —
              all powered by MediaPipe.
            </p>

            <div className="flex flex-col w-full gap-4 sm:flex-row sm:w-auto">
              <button
                onClick={onOpenDemo}
                className="group flex justify-center items-center gap-2.5 px-7 py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 to-cyan-400 text-[#0a0e1a] font-semibold text-sm hover:shadow-lg hover:shadow-cyan-500/25 transition-all duration-300 hover:-translate-y-0.5 w-full sm:w-auto"
              >
                <Camera className="w-4.5 h-4.5" />
                Try Live Demo
                <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
              </button>
              <div className="flex w-full gap-4 sm:w-auto">
                <a
                  href="#quickstart"
                  className="group flex-1 sm:flex-none flex justify-center items-center gap-2 px-2 sm:px-7 py-3.5 rounded-xl border border-white/10 text-white text-sm font-medium hover:border-cyan-500/30 hover:bg-white/5 transition-all duration-300"
                >
                  Get Started
                </a>
                <a
                  href="#how-it-works"
                  className="group flex-1 sm:flex-none flex justify-center items-center gap-2 px-2 sm:px-7 py-3.5 rounded-xl border border-white/10 text-white text-sm font-medium hover:border-cyan-500/30 hover:bg-white/5 transition-all duration-300"
                >
                  <Play className="w-4 h-4 text-cyan-400" />
                  How It Works
                </a>
              </div>
            </div>

            {/* Stats */}
            <div className="flex gap-10 pt-4">
              {[
                { value: '30+', label: 'FPS Real-time' },
                { value: '5', label: 'Privacy Modes' },
                { value: '100%', label: 'Offline Ready' },
              ].map((s) => (
                <div key={s.label}>
                  <div className="text-2xl font-bold text-white">{s.value}</div>
                  <div className="text-xs text-slate-500 mt-0.5">{s.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Right — Mockup */}
          <div className="relative animate-fade-in-up delay-300 opacity-0 lg:flex justify-center hidden">
            <div className="relative">
              {/* Glow ring */}
              <div className="absolute -inset-4 rounded-3xl bg-gradient-to-r from-cyan-500/20 via-violet-500/10 to-cyan-500/20 blur-2xl" />

              <div className="relative rounded-2xl overflow-hidden border border-white/10 shadow-2xl shadow-cyan-500/5 animate-pulse-glow">
                <img
                  src="/images/hero-mockup.png"
                  alt="GesturGuard webcam privacy demo"
                  className="w-full max-w-[520px] object-cover"
                />
                {/* Overlay HUD elements */}
                <div className="absolute top-4 left-4 flex gap-2">
                  <span className="px-2.5 py-1 rounded-md bg-green-500/20 text-green-400 text-[10px] font-bold tracking-wider uppercase border border-green-500/20">
                    ● LIVE
                  </span>
                  <span className="px-2.5 py-1 rounded-md bg-cyan-500/20 text-cyan-400 text-[10px] font-bold tracking-wider uppercase border border-cyan-500/20">
                    BLUR ON
                  </span>
                </div>
                <div className="absolute bottom-4 right-4">
                  <span className="px-2.5 py-1 rounded-md bg-violet-500/20 text-violet-400 text-[10px] font-bold tracking-wider uppercase border border-violet-500/20">
                    🖐️ Gesture Detected
                  </span>
                </div>

                {/* Clickable overlay to open demo */}
                <button
                  onClick={onOpenDemo}
                  className="absolute inset-0 flex items-center justify-center bg-black/30 opacity-0 hover:opacity-100 transition-all duration-300 cursor-pointer group"
                >
                  <div className="w-16 h-16 rounded-full bg-cyan-500/30 border-2 border-cyan-400 flex items-center justify-center backdrop-blur-sm group-hover:scale-110 transition-transform">
                    <Camera className="w-7 h-7 text-cyan-400" />
                  </div>
                </button>
              </div>

              {/* "Click to try" hint */}
              <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-[11px] text-slate-500 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-pulse" />
                Click image or button to try the live demo
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 animate-bounce">
        <span className="text-[10px] text-slate-500 uppercase tracking-widest">Scroll</span>
        <div className="w-5 h-8 rounded-full border border-slate-600 flex items-start justify-center p-1.5">
          <div className="w-1 h-2 bg-cyan-400 rounded-full" />
        </div>
      </div>
    </section>
  );
}
