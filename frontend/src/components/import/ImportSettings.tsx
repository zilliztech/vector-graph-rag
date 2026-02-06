import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Input } from '@/components/ui/input'

interface ImportSettingsProps {
  settings: {
    chunkDocuments: boolean
    chunkSize: number
    extractTriplets: boolean
  }
  onSettingsChange: (settings: ImportSettingsProps['settings']) => void
}

export function ImportSettings({ settings, onSettingsChange }: ImportSettingsProps) {
  return (
    <div className="space-y-4 p-4 border rounded-lg bg-accent/20">
      <h4 className="text-sm font-medium">Import Settings</h4>

      <div className="flex items-center justify-between">
        <div className="space-y-0.5">
          <Label htmlFor="chunk-documents">Chunk documents</Label>
          <p className="text-xs text-muted-foreground">
            Split large documents into smaller chunks
          </p>
        </div>
        <Switch
          id="chunk-documents"
          checked={settings.chunkDocuments}
          onCheckedChange={(checked) =>
            onSettingsChange({ ...settings, chunkDocuments: checked })
          }
        />
      </div>

      {settings.chunkDocuments && (
        <div className="space-y-2">
          <Label htmlFor="chunk-size">Chunk size (characters)</Label>
          <Input
            id="chunk-size"
            type="number"
            min={100}
            max={5000}
            value={settings.chunkSize}
            onChange={(e) =>
              onSettingsChange({
                ...settings,
                chunkSize: parseInt(e.target.value) || 1000,
              })
            }
          />
        </div>
      )}

      <div className="flex items-center justify-between">
        <div className="space-y-0.5">
          <Label htmlFor="extract-triplets">Extract triplets</Label>
          <p className="text-xs text-muted-foreground">
            Use LLM to extract knowledge triplets
          </p>
        </div>
        <Switch
          id="extract-triplets"
          checked={settings.extractTriplets}
          onCheckedChange={(checked) =>
            onSettingsChange({ ...settings, extractTriplets: checked })
          }
        />
      </div>
    </div>
  )
}
