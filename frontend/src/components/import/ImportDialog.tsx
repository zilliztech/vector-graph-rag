import { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { FileUploader } from './FileUploader'
import { UrlInput } from './UrlInput'
import { ImportSettings } from './ImportSettings'
import { ImportProgress } from './ImportProgress'
import { GraphSelector } from './GraphSelector'
import { CreateGraphDialog } from './CreateGraphDialog'
import { GraphStatsPreview } from './GraphStatsPreview'
import { uploadFiles, importDocuments } from '@/api/client'
import { useToast } from '@/hooks/use-toast'
import { useGraphs } from '@/api/queries'

interface ImportDialogProps {
  open: boolean
  onClose: () => void
  onImportComplete?: () => void
}

interface ImportSettings {
  chunkDocuments: boolean
  chunkSize: number
  extractTriplets: boolean
}

export function ImportDialog({ open, onClose, onImportComplete }: ImportDialogProps) {
  const [mode, setMode] = useState<'file' | 'url'>('file')
  const [files, setFiles] = useState<File[]>([])
  const [urls, setUrls] = useState<string[]>([])
  const [settings, setSettings] = useState<ImportSettings>({
    chunkDocuments: true,
    chunkSize: 1000,
    extractTriplets: true,
  })
  const [isImporting, setIsImporting] = useState(false)
  const [progress, setProgress] = useState(0)
  const [selectedGraph, setSelectedGraph] = useState<string>('default')
  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const { toast } = useToast()
  const { data: graphsData } = useGraphs()

  const existingGraphs = graphsData?.graphs.map((g) => g.name) || []

  const handleImport = async () => {
    if ((mode === 'file' && files.length === 0) || (mode === 'url' && urls.length === 0)) {
      toast({
        title: 'No sources',
        description: 'Please add files or URLs to import',
        variant: 'destructive',
      })
      return
    }

    setIsImporting(true)
    setProgress(0)

    try {
      let result

      if (mode === 'file') {
        // Upload files
        setProgress(20)
        result = await uploadFiles(files, {
          chunkDocuments: settings.chunkDocuments,
          chunkSize: settings.chunkSize,
          extractTriplets: settings.extractTriplets,
          graphName: selectedGraph,
        })
      } else {
        // Import URLs
        setProgress(20)
        result = await importDocuments({
          sources: urls,
          chunk_documents: settings.chunkDocuments,
          chunk_size: settings.chunkSize,
          chunk_overlap: 200,
          extract_triplets: settings.extractTriplets,
          graph_name: selectedGraph,
        })
      }

      setProgress(100)

      if (result.success) {
        toast({
          title: 'Import successful',
          description: `Imported to "${selectedGraph}": ${result.num_chunks} chunks, ${result.num_entities} entities, ${result.num_relations} relations`,
        })

        // Reset form
        setFiles([])
        setUrls([])
        onImportComplete?.()
        onClose()
      } else {
        toast({
          title: 'Import failed',
          description: result.errors?.join(', ') || 'Unknown error',
          variant: 'destructive',
        })
      }
    } catch (error) {
      console.error('Import error:', error)
      toast({
        title: 'Import failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    } finally {
      setIsImporting(false)
      setProgress(0)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Import Documents</DialogTitle>
          <DialogDescription>
            Import text documents from files or URLs into the knowledge graph
          </DialogDescription>
        </DialogHeader>

        {/* Graph Selector */}
        <GraphSelector
          value={selectedGraph}
          onChange={setSelectedGraph}
          onCreateNew={() => setShowCreateDialog(true)}
        />

        {/* Graph Stats Preview */}
        <GraphStatsPreview graphName={selectedGraph} />

        <div className="border-t border-slate-200 my-2" />

        <Tabs value={mode} onValueChange={(v) => setMode(v as 'file' | 'url')}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="file">Upload Files</TabsTrigger>
            <TabsTrigger value="url">Import URLs</TabsTrigger>
          </TabsList>

          <TabsContent value="file" className="space-y-4 mt-4">
            <FileUploader
              files={files}
              onFilesChange={setFiles}
              accept=".pdf,.doc,.docx,.txt,.md,.html,.htm"
            />
          </TabsContent>

          <TabsContent value="url" className="space-y-4 mt-4">
            <UrlInput urls={urls} onUrlsChange={setUrls} />
          </TabsContent>
        </Tabs>

        <ImportSettings settings={settings} onSettingsChange={setSettings} />

        {isImporting && <ImportProgress progress={progress} />}

        <div className="flex justify-end gap-2 mt-4">
          <Button variant="outline" onClick={onClose} disabled={isImporting}>
            Cancel
          </Button>
          <Button
            onClick={handleImport}
            disabled={
              isImporting ||
              (mode === 'file' && files.length === 0) ||
              (mode === 'url' && urls.length === 0)
            }
          >
            {isImporting ? 'Importing...' : 'Import'}
          </Button>
        </div>
      </DialogContent>

      {/* Create Graph Dialog */}
      <CreateGraphDialog
        open={showCreateDialog}
        onClose={() => setShowCreateDialog(false)}
        onCreate={(name) => {
          setSelectedGraph(name)
          setShowCreateDialog(false)
          toast({
            title: 'Knowledge base created',
            description: `Created new knowledge base: ${name}`,
          })
        }}
        existingGraphs={existingGraphs}
      />
    </Dialog>
  )
}
