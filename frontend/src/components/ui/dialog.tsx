import { useEffect } from 'react'
import type { ReactNode } from 'react'
import { cn } from '@/utils/cn'

interface DialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  children: ReactNode
}

export function Dialog({ open, onOpenChange, children }: DialogProps) {
  useEffect(() => {
    if (open) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => {
      document.body.style.overflow = ''
    }
  }, [open])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="fixed inset-0 bg-black/50"
        onClick={() => onOpenChange(false)}
      />
      <div className="relative z-50">{children}</div>
    </div>
  )
}

export function DialogContent({
  children,
  className,
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        'bg-white rounded-lg shadow-lg p-6 w-full relative',
        className
      )}
    >
      {children}
    </div>
  )
}

export function DialogHeader({ children }: { children: ReactNode }) {
  return <div className="mb-4">{children}</div>
}

export function DialogTitle({ children }: { children: ReactNode }) {
  return <h2 className="text-lg font-semibold">{children}</h2>
}

export function DialogDescription({ children }: { children: ReactNode }) {
  return <p className="text-sm text-slate-600 mt-1">{children}</p>
}
