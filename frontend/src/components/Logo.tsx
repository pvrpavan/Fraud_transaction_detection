export function Logo({ className = 'w-8 h-8' }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <defs>
        <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#6366F1" />
          <stop offset="100%" stopColor="#A855F7" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="14" fill="#0F172A" />
      <path
        d="M32 8 L52 16 L52 34 C52 44 44 52 32 56 C20 52 12 44 12 34 L12 16 Z"
        fill="url(#logoGrad)"
        opacity="0.9"
      />
      <path
        d="M32 13 L48 19.5 L48 33 C48 41.5 41 48.5 32 52 C23 48.5 16 41.5 16 33 L16 19.5 Z"
        fill="#1E1B4B"
        opacity="0.85"
      />
      <path
        d="M24 33 L30 39 L42 25"
        stroke="#34D399"
        strokeWidth="4"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}

export function LogoFull({ className = '' }: { className?: string }) {
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <div className="p-1.5 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-600/20 border border-indigo-500/20 shadow-lg shadow-indigo-500/10">
        <Logo className="w-7 h-7" />
      </div>
      <div>
        <h1 className="text-lg font-bold text-white tracking-tight">
          Fraud<span className="gradient-text">Guard</span>
        </h1>
        <p className="text-[10px] text-gray-500 font-medium tracking-wider uppercase -mt-0.5">
          Intelligent Transaction Monitoring
        </p>
      </div>
    </div>
  );
}
