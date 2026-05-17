import { Camera, Hand, ShieldCheck } from 'lucide-react';
import { useInView } from '../hooks/useInView';

const steps = [
  {
    icon: Camera,
    step: '01',
    title: 'Launch & Allow Webcam',
    description:
      'Start the application and grant camera access. GesturGuard processes everything locally — your video never leaves your device.',
  },
  {
    icon: Hand,
    step: '02',
    title: 'Use Hand Gestures',
    description:
      'Raise your hand to toggle blur, make a fist for pixelation, peace sign for custom background, or thumbs-up for sign-language mode.',
  },
  {
    icon: ShieldCheck,
    step: '03',
    title: 'Enjoy Real-Time Privacy',
    description:
      'All effects are applied instantly at 30+ FPS. Switch modes on the fly and stay protected throughout your session.',
  },
];

export default function HowItWorks() {
  const { ref, inView } = useInView(0.1);

  return (
    <section id="how-it-works" className="relative py-28">
      {/* bg */}
      <div className="absolute inset-0 bg-gradient-to-b from-dark-900 via-dark-800/50 to-dark-900" />

      <div ref={ref} className="relative mx-auto max-w-7xl px-6">
        {/* Header */}
        <div className={`text-center max-w-2xl mx-auto mb-20 ${inView ? 'animate-fade-in-up' : 'opacity-0'}`}>
          <span className="inline-block px-4 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-medium tracking-wide uppercase mb-4">
            How It Works
          </span>
          <h2 className="text-4xl sm:text-5xl font-extrabold text-white mb-4 tracking-tight">
            Three Steps. Total Privacy.
          </h2>
          <p className="text-slate-400 text-lg">
            No complex setup, no cloud dependency — just install, gesture, and go.
          </p>
        </div>

        {/* Steps */}
        <div className="grid md:grid-cols-3 gap-8 relative">
          {/* Connector line (desktop) */}
          <div className="hidden md:block absolute top-20 left-[16.7%] right-[16.7%] h-px bg-gradient-to-r from-cyan-500/30 via-violet-500/30 to-cyan-500/30" />

          {steps.map((s, i) => (
            <div
              key={s.step}
              className={`relative text-center ${inView ? 'animate-fade-in-up' : 'opacity-0'}`}
              style={{ animationDelay: `${i * 0.2}s` }}
            >
              {/* Circle */}
              <div className="relative mx-auto w-20 h-20 mb-6">
                <div className="absolute inset-0 rounded-full bg-cyan-500/10 border border-cyan-500/20" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <s.icon className="w-8 h-8 text-cyan-400" />
                </div>
                {/* Step number badge */}
                <div className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-violet-500 flex items-center justify-center">
                  <span className="text-[11px] font-bold text-white">{s.step}</span>
                </div>
              </div>

              <h3 className="text-xl font-bold text-white mb-3">{s.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed max-w-xs mx-auto">
                {s.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
