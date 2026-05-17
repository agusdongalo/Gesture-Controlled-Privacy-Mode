import { Copy, Check } from 'lucide-react';
import { useState } from 'react';
import { useInView } from '../hooks/useInView';

const commands = [
  { comment: '# Clone the repository', cmd: 'git clone https://github.com/your-username/gesturguard.git' },
  { comment: '# Navigate to the project', cmd: 'cd gesturguard' },
  { comment: '# Install dependencies', cmd: 'pip install -r requirements.txt' },
  { comment: '# Run the application', cmd: 'python main.py' },
];

export default function QuickStart() {
  const [copied, setCopied] = useState(false);
  const { ref, inView } = useInView(0.1);

  const allCmds = commands.map((c) => c.cmd).join('\n');

  const handleCopy = () => {
    navigator.clipboard.writeText(allCmds);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section id="quickstart" className="relative py-28">
      <div className="absolute inset-0 bg-gradient-to-b from-dark-900 via-dark-800/50 to-dark-900" />

      <div ref={ref} className="relative mx-auto max-w-3xl px-6">
        {/* Header */}
        <div className={`text-center mb-12 ${inView ? 'animate-fade-in-up' : 'opacity-0'}`}>
          <span className="inline-block px-4 py-1.5 rounded-full bg-pink-500/10 border border-pink-500/20 text-pink-400 text-xs font-medium tracking-wide uppercase mb-4">
            Quick Start
          </span>
          <h2 className="text-4xl sm:text-5xl font-extrabold text-white mb-4 tracking-tight">
            Up and Running in Seconds
          </h2>
          <p className="text-slate-400 text-lg">
            Four commands. That's all it takes.
          </p>
        </div>

        {/* Terminal */}
        <div className={`terminal rounded-2xl overflow-hidden ${inView ? 'animate-fade-in-up delay-200' : 'opacity-0'}`}>
          {/* Title bar */}
          <div className="flex items-center gap-2 px-5 py-3.5 bg-dark-700/50 border-b border-white/5">
            <div className="terminal-dot bg-red-500/80" />
            <div className="terminal-dot bg-yellow-500/80" />
            <div className="terminal-dot bg-green-500/80" />
            <span className="ml-3 text-xs text-slate-500">terminal</span>
            <button
              onClick={handleCopy}
              className="ml-auto flex items-center gap-1.5 px-3 py-1 rounded-md text-xs text-slate-400 hover:text-cyan-400 hover:bg-white/5 transition-all"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5" /> Copied!
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" /> Copy
                </>
              )}
            </button>
          </div>

          {/* Commands */}
          <div className="px-6 py-5 space-y-3 text-sm">
            {commands.map((c, i) => (
              <div key={i}>
                <div className="text-slate-600 text-xs">{c.comment}</div>
                <div className="flex gap-2">
                  <span className="text-cyan-400 select-none">$</span>
                  <span className="text-slate-300">{c.cmd}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Extra note */}
        <p className={`text-center text-sm text-slate-500 mt-6 ${inView ? 'animate-fade-in-up delay-300' : 'opacity-0'}`}>
          Requires <span className="text-slate-400">Python 3.8+</span> and a webcam.
          Works on <span className="text-slate-400">macOS, Windows & Linux</span>.
        </p>
      </div>
    </section>
  );
}
