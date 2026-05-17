import { useInView } from '../hooks/useInView';

const techs = [
  {
    name: 'Python',
    icon: '🐍',
    description: 'Core application language',
  },
  {
    name: 'MediaPipe',
    icon: '🤖',
    description: 'Hand & face detection ML',
  },
  {
    name: 'OpenCV',
    icon: '📷',
    description: 'Computer vision processing',
  },
  {
    name: 'NumPy',
    icon: '🔢',
    description: 'High-perf array operations',
  },
  {
    name: 'Pillow',
    icon: '🖼️',
    description: 'Image processing library',
  },
  {
    name: 'Tkinter',
    icon: '🖥️',
    description: 'GUI framework',
  },
];

export default function TechStack() {
  const { ref, inView } = useInView(0.1);

  return (
    <section id="tech-stack" className="relative py-28 overflow-hidden">
      <div className="absolute bottom-0 right-0 w-[500px] h-[400px] bg-cyan-500/5 rounded-full blur-[120px]" />

      <div ref={ref} className="relative mx-auto max-w-7xl px-6">
        {/* Header */}
        <div className={`text-center max-w-2xl mx-auto mb-16 ${inView ? 'animate-fade-in-up' : 'opacity-0'}`}>
          <span className="inline-block px-4 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium tracking-wide uppercase mb-4">
            Tech Stack
          </span>
          <h2 className="text-4xl sm:text-5xl font-extrabold text-white mb-4 tracking-tight">
            Built With the Best
          </h2>
          <p className="text-slate-400 text-lg">
            Industry-standard tools for computer vision and machine learning.
          </p>
        </div>

        {/* Tech grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-5">
          {techs.map((t, i) => (
            <div
              key={t.name}
              className={`glass-card rounded-2xl p-5 text-center ${
                inView ? 'animate-fade-in-up' : 'opacity-0'
              }`}
              style={{ animationDelay: `${i * 0.08}s` }}
            >
              <div className="text-3xl mb-3">{t.icon}</div>
              <h3 className="text-sm font-semibold text-white mb-1">{t.name}</h3>
              <p className="text-[11px] text-slate-500">{t.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
