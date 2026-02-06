import { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { AlertCircle, Info } from 'lucide-react'

interface CreateGraphDialogProps {
  open: boolean
  onClose: () => void
  onCreate: (graphName: string) => void
  existingGraphs: string[]
}

export function CreateGraphDialog({
  open,
  onClose,
  onCreate,
  existingGraphs,
}: CreateGraphDialogProps) {
  const [graphName, setGraphName] = useState('')
  const [error, setError] = useState<string | null>(null)

  const validateGraphName = (name: string): string | null => {
    // Empty check
    if (!name.trim()) {
      return 'Name is required'
    }

    // Format check (only letters, numbers, underscores, hyphens)
    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
      return 'Only letters, numbers, underscores (_), and hyphens (-) are allowed'
    }

    // Length check
    if (name.length > 50) {
      return 'Name is too long (maximum 50 characters)'
    }

    if (name.length < 2) {
      return 'Name is too short (minimum 2 characters)'
    }

    // Duplicate check
    if (existingGraphs.includes(name)) {
      return 'A knowledge base with this name already exists'
    }

    return null
  }

  const handleCreate = () => {
    const validationError = validateGraphName(graphName)
    if (validationError) {
      setError(validationError)
      return
    }

    onCreate(graphName)
    setGraphName('')
    setError(null)
  }

  const handleClose = () => {
    setGraphName('')
    setError(null)
    onClose()
  }

  const handleNameChange = (value: string) => {
    setGraphName(value)
    // Clear error on change
    if (error) {
      setError(null)
    }
  }

  // Real-time validation for display
  const currentError = graphName ? validateGraphName(graphName) : null
  const isValid = graphName && !currentError

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Create New Knowledge Base</DialogTitle>
          <DialogDescription>
            Create a new knowledge base to organize your documents by project or topic
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Name input */}
          <div className="space-y-2">
            <Label htmlFor="graph-name">Knowledge Base Name</Label>
            <Input
              id="graph-name"
              placeholder="e.g., my_project, research_papers"
              value={graphName}
              onChange={(e) => handleNameChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && isValid) {
                  handleCreate()
                }
              }}
              className={cn(
                error || currentError ? 'border-red-300 focus:border-red-500 focus:ring-red-500' : ''
              )}
            />
            {(error || currentError) && (
              <div className="flex items-start gap-2 text-sm text-red-600">
                <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span>{error || currentError}</span>
              </div>
            )}
            {!error && !currentError && graphName && (
              <div className="text-xs text-slate-500">
                💡 Only letters, numbers, underscores (_), and hyphens (-) allowed
              </div>
            )}
          </div>

          {/* Preview */}
          {graphName && !currentError && (
            <div className="rounded-lg bg-blue-50 border border-blue-200 p-3 space-y-2">
              <div className="flex items-start gap-2 text-sm text-blue-900">
                <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <div>
                  <div className="font-medium mb-1">This will create 3 collections:</div>
                  <ul className="space-y-0.5 text-xs font-mono text-blue-700">
                    <li>• {graphName}_vgrag_entities</li>
                    <li>• {graphName}_vgrag_relations</li>
                    <li>• {graphName}_vgrag_passages</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={handleClose}>
            Cancel
          </Button>
          <Button onClick={handleCreate} disabled={!isValid}>
            Create
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

// Helper function (should be imported from utils but defined here for completeness)
function cn(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(' ')
}
