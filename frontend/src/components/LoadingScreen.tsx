import { Logo } from './Logo';
import { Shield } from 'lucide-react';

export function LoadingScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center mesh-gradient">
      <div className="text-center">
        <div className="relative mb-8">
          <div className="animate-pulse">
            <Logo className="w-24 h-24 mx-auto" />
          </div>
          <div className="absolute inset-0 w-24 h-24 mx-auto rounded-2xl bg-indigo-500/20 animate-ping" />
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">
          Fraud<span className="gradient-text">Guard</span>
        </h2>
        <p className="text-sm text-gray-400 mb-1">Intelligent Transaction Monitoring</p>
        <div className="flex items-center justify-center gap-2 mt-4 mb-6">
          <Shield className="w-3.5 h-3.5 text-indigo-400" />
          <p className="text-xs text-gray-500">Initializing fraud detection engine...</p>
        </div>
        <div className="w-48 mx-auto h-1 bg-gray-800 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full shimmer" style={{ width: '70%' }} />
        </div>
        <div className="mt-6 flex justify-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: '0ms' }} />
          <div className="w-2 h-2 rounded-full bg-purple-500 animate-bounce" style={{ animationDelay: '150ms' }} />
          <div className="w-2 h-2 rounded-full bg-pink-500 animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    </div>
  );
}
