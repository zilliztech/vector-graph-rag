import { useState, useCallback } from 'react'
import { Search, Loader2, X } from 'lucide-react'
import { cn } from '@/utils/cn'
import { useSearchStore } from '@/stores/searchStore'
import { useGraphStore } from '@/stores/graphStore'
import { useSearch } from '@/api/queries'

function SearchInput() {
  const [inputValue, setInputValue] = useState('')
  const { setQuery, setSearching, setResult, setError, reset } = useSearchStore()
  const { setSubgraph, clearGraph } = useGraphStore()

  const searchMutation = useSearch()

  const handleSearch = useCallback(async () => {
    const query = inputValue.trim()
    if (!query || searchMutation.isPending) return

    setQuery(query)
    setSearching(true)
    setError(null)
    clearGraph()

    try {
      const result = await searchMutation.mutateAsync({ question: query })
      setResult(result)
      setSubgraph(result.subgraph, result.retrieval_detail, result.rerank_result)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Search failed'
      setError(message)
    } finally {
      setSearching(false)
    }
  }, [inputValue, searchMutation, setQuery, setSearching, setResult, setError, setSubgraph, clearGraph])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSearch()
      }
    },
    [handleSearch]
  )

  const handleClear = useCallback(() => {
    setInputValue('')
    reset()
    clearGraph()
  }, [reset, clearGraph])

  const isLoading = searchMutation.isPending

  return (
    <div className="relative w-full max-w-2xl mx-auto">
      <div className="relative">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter your question..."
          disabled={isLoading}
          className={cn(
            'w-full px-4 py-3 pr-24',
            'border border-slate-300 rounded-lg',
            'focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500',
            'text-slate-700 placeholder:text-slate-400',
            'transition-all duration-200',
            'disabled:opacity-60 disabled:cursor-not-allowed'
          )}
        />

        {/* Clear button */}
        {inputValue && !isLoading && (
          <button
            onClick={handleClear}
            className={cn(
              'absolute right-14 top-1/2 -translate-y-1/2',
              'p-1.5 rounded-md',
              'text-slate-400 hover:text-slate-600 hover:bg-slate-100',
              'transition-colors'
            )}
            title="Clear"
          >
            <X className="w-4 h-4" />
          </button>
        )}

        {/* Search button */}
        <button
          onClick={handleSearch}
          disabled={isLoading || !inputValue.trim()}
          className={cn(
            'absolute right-2 top-1/2 -translate-y-1/2',
            'p-2 rounded-md',
            'text-slate-500 hover:text-slate-700 hover:bg-slate-100',
            'disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent',
            'transition-colors'
          )}
          title="Search"
        >
          {isLoading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Search className="w-5 h-5" />
          )}
        </button>
      </div>

      {/* Query entities display */}
      <QueryEntitiesDisplay />
    </div>
  )
}

function QueryEntitiesDisplay() {
  const { result, isSearching } = useSearchStore()

  if (isSearching) {
    return (
      <div className="mt-2 flex items-center gap-2 text-sm text-slate-500">
        <Loader2 className="w-3 h-3 animate-spin" />
        <span>Analyzing query...</span>
      </div>
    )
  }

  if (!result?.query_entities?.length) return null

  return (
    <div className="mt-2 flex flex-wrap items-center gap-2">
      <span className="text-xs text-slate-500">Entities:</span>
      {result.query_entities.map((entity, index) => (
        <span
          key={index}
          className="px-2 py-0.5 text-xs rounded-full bg-blue-50 text-blue-700 border border-blue-200"
        >
          {entity}
        </span>
      ))}
    </div>
  )
}

export default SearchInput
