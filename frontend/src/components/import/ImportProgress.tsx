import { Progress } from '@/components/ui/progress'
import { Loader2 } from 'lucide-react'

interface ImportProgressProps {
  progress: number
}

export function ImportProgress({ progress }: ImportProgressProps) {
  return (
    <div className="space-y-2 p-4 border rounded-lg bg-accent/20">
      <div className="flex items-center gap-2">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm font-medium">Importing documents...</span>
      </div>
      <Progress value={progress} className="w-full" />
      <p className="text-xs text-muted-foreground">{progress}% complete</p>
    </div>
  )
}
