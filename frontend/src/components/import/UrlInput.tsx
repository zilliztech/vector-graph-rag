import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Link, X } from 'lucide-react'

interface UrlInputProps {
  urls: string[]
  onUrlsChange: (urls: string[]) => void
}

export function UrlInput({ urls, onUrlsChange }: UrlInputProps) {
  const [inputValue, setInputValue] = useState('')
  const [error, setError] = useState('')

  const addUrl = () => {
    const url = inputValue.trim()
    if (!url) return

    try {
      new URL(url)
      if (urls.includes(url)) {
        setError('URL already added')
        return
      }
      onUrlsChange([...urls, url])
      setInputValue('')
      setError('')
    } catch {
      setError('Invalid URL format')
    }
  }

  const removeUrl = (index: number) => {
    onUrlsChange(urls.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <Input
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value)
            setError('')
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              addUrl()
            }
          }}
          placeholder="https://example.com/article"
          className={error ? 'border-destructive' : ''}
        />
        <Button type="button" onClick={addUrl}>
          Add
        </Button>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {urls.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">URLs to import ({urls.length})</p>
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {urls.map((url, i) => (
              <div
                key={i}
                className="flex items-center gap-2 p-2 rounded bg-accent/50 text-sm"
              >
                <Link className="w-4 h-4 flex-shrink-0" />
                <span className="flex-1 truncate">{url}</span>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => removeUrl(i)}
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
