import { useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Upload, X, File } from 'lucide-react'
import { cn } from '@/lib/utils'

interface FileUploaderProps {
  files: File[]
  onFilesChange: (files: File[]) => void
  accept?: string
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`
}

export function FileUploader({ files, onFilesChange, accept }: FileUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (newFiles: FileList | null) => {
    if (!newFiles) return
    onFilesChange([...files, ...Array.from(newFiles)])
  }

  const removeFile = (index: number) => {
    onFilesChange(files.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-4">
      <div
        className={cn(
          'border-2 border-dashed rounded-lg p-8 text-center transition-colors',
          'hover:border-primary/50 hover:bg-accent/50'
        )}
        onDrop={(e) => {
          e.preventDefault()
          handleFileSelect(e.dataTransfer.files)
        }}
        onDragOver={(e) => e.preventDefault()}
      >
        <Upload className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
        <p className="text-sm text-muted-foreground mb-2">
          Drag and drop files here, or click to browse
        </p>
        <p className="text-xs text-muted-foreground mb-4">
          Supported: PDF, DOCX, TXT, MD, HTML
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={accept}
          onChange={(e) => handleFileSelect(e.target.files)}
          className="hidden"
        />
        <Button
          type="button"
          variant="outline"
          onClick={() => inputRef.current?.click()}
        >
          Browse Files
        </Button>
      </div>

      {files.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Selected files ({files.length})</p>
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {files.map((file, i) => (
              <div
                key={i}
                className="flex items-center gap-2 p-2 rounded bg-accent/50 text-sm"
              >
                <File className="w-4 h-4 flex-shrink-0" />
                <span className="flex-1 truncate">{file.name}</span>
                <span className="text-xs text-muted-foreground">
                  {formatFileSize(file.size)}
                </span>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => removeFile(i)}
                  className="h-6 w-6 p-0"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
