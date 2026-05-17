import { Hand, Scan, Image, MessageSquare, Eye } from 'lucide-react';
import { useInView } from '../hooks/useInView';

const features = [
  {
    icon: Hand,
    title: 'Gesture Control',
    description:
      'Toggle any privacy mode with intuitive hand gestures. No keyboard, no clicks — just raise your hand.',
    color: 'cyan',
  },
  {
    icon: Eye,
    title: 'Dynamic Background Blur',
    description:
      'Real-time depth-aware background blur that keeps you in focus while hiding your surroundings.',
    color: 'violet',
  },
  {
    icon: Scan,
    title: 'Face & Hand Pixelation',
    description:
      'Instantly anonymize faces and hands in the frame with configurable pixelation intensity.',
    color: 'pink',
  },
  {
    icon: Image,
    title: 'Custom Backgrounds',
    description:
      'Replace your real background with any image. Perfect for virtual meetings and streaming.',
    color: 'amber',
  },
  {
    icon: MessageSquare,
    title: 'Sign Language Mode',
    description:
      'Built-in sign language recognition that translates ASL gestures to on-screen text in real time.',
    color: 'emerald',
  },
];

const colorMap: Record<string, { bg: string; text: string; border: string; glow: string }> = {
  cyan: {
    bg: 'bg-cyan-500/10',
    text: 'text-cyan-400',
    border: 'border-cyan-500/20',
    glow: 'group-hover:shadow-cyan-500/10',
  },
  violet: {
    bg: 'bg-violet-500/10',
    text: 'text-violet-400',
    border: 'border-violet-500/20',
    glow: 'group-hover:shadow-violet-500/10',
  },
  pink: {
    bg: 'bg-pink-500/10',
    text: 'text-pink-400',
    border: 'border-pink-500/20',
    glow: 'group-hover:shadow-pink-500/10',
  },
  amber: {
    bg: 'bg-amber-500/10',
    text: 'text-amber-400',
    border: 'border-amber-500/20',
    glow: 'group-hover:shadow-amber-500/10',
  },
  emerald: {
    bg: 'bg-emerald-500/10',
    text: 'text-emerald-400',
    border: 'border-emerald-500/20',
    glow: 'group-hover:shadow-emerald-500/10',
  },
};

export default function Features() {
  const { ref, inView } = useInView(0.1);

  return (
    <section id="features" className="relative py-28 overflow-hidden">
      {/* bg glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-violet-500/5 rounded-full blur-[120px]" />

      <div ref={ref} className="relative mx-auto max-w-7xl px-6">
        {/* Section header */}
        <div className={`text-center max-w-2xl mx-auto mb-16 ${inView ? 'animate-fade-in-up' : 'opacity-0'}`}>
          <span className="inline-block px-4 py-1.5 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400 text-xs font-medium tracking-wide uppercase mb-4">
            Features
          </span>
          <h2 className="text-4xl sm:text-5xl font-extrabold text-white mb-4 tracking-tight">
            Every Privacy Tool You Need
          </h2>
          <p className="text-slate-400 text-lg">
            Five powerful modes, all controllable with simple hand gestures.
            No extra hardware required.
          </p>
        </div>

        {/* Feature grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((f, i) => {
            const c = colorMap[f.color];
            return (
              <div
                key={f.title}
                className={`group glass-card rounded-2xl p-7 ${
                  inView ? 'animate-fade-in-up' : 'opacity-0'
                }`}
                style={{ animationDelay: `${i * 0.1}s` }}
              >
                <div
                  className={`inline-flex items-center justify-center w-12 h-12 rounded-xl ${c.bg} ${c.border} border mb-5`}
                >
                  <f.icon className={`w-6 h-6 ${c.text}`} />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">{f.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed">{f.description}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
