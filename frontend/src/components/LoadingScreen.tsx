import { Logo } from './Logo';

export function LoadingScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <div className="relative mb-8">
          <div className="animate-pulse">
            <Logo className="w-20 h-20 mx-auto" />
          </div>
          <div className="absolute inset-0 w-20 h-20 mx-auto rounded-2xl bg-indigo-500/20 animate-ping" />
        </div>
        <h2 className="text-xl font-bold text-white mb-2">FraudGuard AI</h2>
        <p className="text-sm text-gray-400">Loading Dashboard...</p>
        <div className="mt-6 flex justify-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: '0ms' }} />
          <div className="w-2 h-2 rounded-full bg-purple-500 animate-bounce" style={{ animationDelay: '150ms' }} />
          <div className="w-2 h-2 rounded-full bg-pink-500 animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    </div>
  );
}
