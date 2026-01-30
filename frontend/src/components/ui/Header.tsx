import { Database, Settings, Github } from 'lucide-react'
import { cn } from '@/utils/cn'

interface HeaderProps {
  graphName?: string
}

function Header({ graphName = 'default' }: HeaderProps) {
  return (
    <header className="h-14 border-b border-slate-200 bg-white px-4 flex items-center justify-between">
      {/* Left section - Logo and title */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-500 flex items-center justify-center">
          <Database className="w-4 h-4 text-white" />
        </div>
        <div>
          <h1 className="text-sm font-semibold text-slate-800">
            Vector Graph RAG
          </h1>
          <p className="text-xs text-slate-500">Knowledge Graph Explorer</p>
        </div>
      </div>

      {/* Center section - Graph selector */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500">Dataset:</span>
        <button
          className={cn(
            'px-3 py-1.5 rounded-md text-sm',
            'border border-slate-200 bg-white',
            'hover:bg-slate-50 transition-colors',
            'flex items-center gap-2'
          )}
        >
          <span className="font-medium text-slate-700">{graphName}</span>
          <svg
            className="w-4 h-4 text-slate-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </button>
      </div>

      {/* Right section - Actions */}
      <div className="flex items-center gap-2">
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 rounded-md text-slate-500 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          title="GitHub"
        >
          <Github className="w-5 h-5" />
        </a>
        <button
          className="p-2 rounded-md text-slate-500 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          title="Settings"
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>
    </header>
  )
}

export default Header
