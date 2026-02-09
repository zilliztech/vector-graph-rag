import { create } from 'zustand'
import type {
  QueryResponse,
  TimelineStep,
  StepStatus,
  ExpansionStep,
} from '@/types/api'

interface SearchState {
  // Query state
  query: string
  isSearching: boolean
  result: QueryResponse | null
  error: string | null

  // Timeline state
  currentStepIndex: number
  timelineSteps: TimelineStep[]

  // Actions
  setQuery: (query: string) => void
  setSearching: (isSearching: boolean) => void
  setResult: (result: QueryResponse | null) => void
  setError: (error: string | null) => void
  setCurrentStep: (index: number) => void
  buildTimeline: (result: QueryResponse) => void
  reset: () => void
}


// Build timeline steps from query result
function buildTimelineSteps(result: QueryResponse): TimelineStep[] {
  const steps: TimelineStep[] = []
  const retrievalDetail = result.retrieval_detail
  const rerankResult = result.rerank_result

  // Get all IDs for reference
  const allEntityIds = result.subgraph.entity_ids
  const allRelationIds = result.subgraph.relation_ids
  const seedEntityIds = retrievalDetail?.entity_ids || []
  const seedRelationIds = retrievalDetail?.relation_ids || []
  const selectedRelationIds = rerankResult?.selected_relation_ids || []

  // Calculate expanded IDs (all minus seeds)
  const seedEntitySet = new Set(seedEntityIds)
  const seedRelationSet = new Set(seedRelationIds)
  const expandedEntityIds = allEntityIds.filter(id => !seedEntitySet.has(id))
  const expandedRelationIds = allRelationIds.filter(id => !seedRelationSet.has(id))

  // Calculate filtered relation IDs (expanded but not selected)
  const selectedSet = new Set(selectedRelationIds)
  const filteredRelationIds = allRelationIds.filter(id => !selectedSet.has(id))

  // Step 1: Retrieve seed nodes
  if (retrievalDetail) {
    steps.push({
      id: 'retrieve-seeds',
      label: 'Retrieve Seeds',
      status: 'completed',
      description: 'Vector search for initial entities and relations',
      nodeStatus: {
        seedEntityIds: [...seedEntityIds],
        seedRelationIds: [...seedRelationIds],
        expandedEntityIds: [],
        expandedRelationIds: [],
        filteredRelationIds: [],
        selectedRelationIds: [],
      },
      stats: {
        entityCount: seedEntityIds.length,
        relationCount: seedRelationIds.length,
      },
    })
  }

  // Step 2+: Expansion steps from history
  const history = result.subgraph.expansion_history || []
  let currentExpandedEntities: string[] = []
  let currentExpandedRelations: string[] = []
  let expandStepNumber = 1

  history.forEach((step: ExpansionStep) => {
    const newEntities = step.added_entity_ids || step.new_entity_ids || []
    const newRelations = step.added_relation_ids || step.new_relation_ids || []

    // Filter out seeds from new entities/relations
    const trulyNewEntities = newEntities.filter(id => !seedEntitySet.has(id))
    const trulyNewRelations = newRelations.filter(id => !seedRelationSet.has(id))

    // Always accumulate IDs (even for init steps)
    currentExpandedEntities = [...currentExpandedEntities, ...trulyNewEntities]
    currentExpandedRelations = [...currentExpandedRelations, ...trulyNewRelations]

    // Skip creating timeline step for initialization operations
    // (init, init_merge, etc.) - they don't need separate visualization
    // but their IDs are already accumulated above
    if (step.operation === 'init' || step.operation === 'init_merge' || step.operation.startsWith('init')) {
      return
    }

    // Calculate visible counts at this step (seeds + expanded so far)
    const visibleEntityCount = seedEntityIds.length + currentExpandedEntities.length
    const visibleRelationCount = seedRelationIds.length + currentExpandedRelations.length

    steps.push({
      id: `expand-${expandStepNumber}`,
      label: 'Expand Subgraph',
      status: 'completed',
      description: step.description || step.operation,
      nodeStatus: {
        seedEntityIds: [...seedEntityIds],
        seedRelationIds: [...seedRelationIds],
        expandedEntityIds: [...currentExpandedEntities],
        expandedRelationIds: [...currentExpandedRelations],
        filteredRelationIds: [],
        selectedRelationIds: [],
      },
      stats: {
        entityCount: visibleEntityCount,
        relationCount: visibleRelationCount,
      },
    })
    expandStepNumber++
  })

  // Step: Eviction Result (only shown when eviction actually occurred)
  if (result.eviction_result?.occurred) {
    steps.push({
      id: 'eviction',
      label: 'Eviction Result',
      status: 'completed',
      description: `Evicted from ${result.eviction_result.before_count} to ${result.eviction_result.after_count} relations`,
      nodeStatus: {
        seedEntityIds: [...seedEntityIds],
        seedRelationIds: [...seedRelationIds],
        expandedEntityIds: [...expandedEntityIds],
        expandedRelationIds: [...expandedRelationIds],
        filteredRelationIds: [...filteredRelationIds],
        selectedRelationIds: [],
      },
      stats: {
        entityCount: allEntityIds.length,
        relationCount: result.eviction_result.after_count,
      },
    })
  }

  // Step: LLM Rerank
  if (rerankResult) {
    steps.push({
      id: 'rerank',
      label: 'LLM Rerank',
      status: 'completed',
      description: `Selected ${selectedRelationIds.length} most relevant relations`,
      nodeStatus: {
        seedEntityIds: [...seedEntityIds],
        seedRelationIds: [...seedRelationIds],
        expandedEntityIds: [...expandedEntityIds],
        expandedRelationIds: [...expandedRelationIds],
        filteredRelationIds: [...filteredRelationIds],
        selectedRelationIds: [...selectedRelationIds],
      },
      stats: {
        entityCount: allEntityIds.length,
        relationCount: selectedRelationIds.length,
      },
    })
  }

  // Step: Generate Answer
  steps.push({
    id: 'generate',
    label: 'Generate Answer',
    status: 'completed',
    description: 'LLM generates answer from retrieved context',
    nodeStatus: {
      seedEntityIds: [...seedEntityIds],
      seedRelationIds: [...seedRelationIds],
      expandedEntityIds: [...expandedEntityIds],
      expandedRelationIds: [...expandedRelationIds],
      filteredRelationIds: [...filteredRelationIds],
      selectedRelationIds: [...selectedRelationIds],
    },
    stats: {
      entityCount: allEntityIds.length,
      relationCount: selectedRelationIds.length,
    },
  })

  return steps
}

// Update step statuses based on current step
function updateStepStatuses(
  steps: TimelineStep[],
  currentIndex: number
): TimelineStep[] {
  return steps.map((step, index) => {
    let status: StepStatus = 'pending'
    if (index < currentIndex) {
      status = 'completed'
    } else if (index === currentIndex) {
      status = 'active'
    }
    return { ...step, status }
  })
}

export const useSearchStore = create<SearchState>((set, get) => ({
  // Initial state
  query: '',
  isSearching: false,
  result: null,
  error: null,
  currentStepIndex: -1,
  timelineSteps: [],

  // Actions
  setQuery: (query) => set({ query }),

  setSearching: (isSearching) => set({ isSearching }),

  setResult: (result) => {
    if (result) {
      const steps = buildTimelineSteps(result)
      set({
        result,
        timelineSteps: updateStepStatuses(steps, steps.length - 1),
        currentStepIndex: steps.length - 1,
        error: null,
      })
    } else {
      set({ result: null, timelineSteps: [], currentStepIndex: -1 })
    }
  },

  setError: (error) => set({ error, isSearching: false }),

  setCurrentStep: (index) => {
    const { timelineSteps } = get()
    if (index >= 0 && index < timelineSteps.length) {
      set({
        currentStepIndex: index,
        timelineSteps: updateStepStatuses(timelineSteps, index),
      })
    }
  },

  buildTimeline: (result) => {
    const steps = buildTimelineSteps(result)
    set({
      timelineSteps: updateStepStatuses(steps, steps.length - 1),
      currentStepIndex: steps.length - 1,
    })
  },

  reset: () =>
    set({
      query: '',
      isSearching: false,
      result: null,
      error: null,
      currentStepIndex: -1,
      timelineSteps: [],
    }),
}))
