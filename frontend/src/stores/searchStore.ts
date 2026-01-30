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

  // Step 1: Extract entities from query
  steps.push({
    id: 'extract-entities',
    label: 'Extract Entities',
    status: 'completed',
    description: `Identified ${result.query_entities.length} entities from query`,
    entityIds: [],
    relationIds: [],
    stats: {
      entityCount: result.query_entities.length,
      relationCount: 0,
    },
  })

  // Step 2: Retrieve seed nodes
  const retrievalDetail = result.retrieval_detail
  if (retrievalDetail) {
    steps.push({
      id: 'retrieve-seeds',
      label: 'Retrieve Seeds',
      status: 'completed',
      description: 'Vector search for initial entities and relations',
      entityIds: retrievalDetail.entity_ids,
      relationIds: retrievalDetail.relation_ids,
      highlightIds: [
        ...retrievalDetail.entity_ids,
        ...retrievalDetail.relation_ids,
      ],
      stats: {
        entityCount: retrievalDetail.entity_ids.length,
        relationCount: retrievalDetail.relation_ids.length,
      },
    })
  }

  // Step 3+: Expansion steps from history
  const history = result.subgraph.expansion_history || []
  let accumulatedEntities: string[] = retrievalDetail?.entity_ids || []
  let accumulatedRelations: string[] = retrievalDetail?.relation_ids || []

  history.forEach((step: ExpansionStep, index: number) => {
    const newEntities = step.added_entity_ids || step.new_entity_ids || []
    const newRelations = step.added_relation_ids || step.new_relation_ids || []

    accumulatedEntities = [...accumulatedEntities, ...newEntities]
    accumulatedRelations = [...accumulatedRelations, ...newRelations]

    steps.push({
      id: `expand-${index}`,
      label: step.operation === 'init' ? 'Initialize Graph' : `Expand (Step ${step.step})`,
      status: 'completed',
      description: step.description || step.operation,
      entityIds: [...accumulatedEntities],
      relationIds: [...accumulatedRelations],
      highlightIds: [...newEntities, ...newRelations],
      stats: {
        entityCount: step.total_entities,
        relationCount: step.total_relations,
      },
    })
  })

  // Step: Eviction (if relations were filtered)
  const expandedCount = result.subgraph.relation_ids.length
  const rerankedCount = result.rerank_result?.selected_relation_ids.length || expandedCount

  if (rerankedCount < expandedCount) {
    steps.push({
      id: 'eviction',
      label: 'Filter Relations',
      status: 'completed',
      description: `Filtered from ${expandedCount} to ${rerankedCount} relations`,
      entityIds: result.subgraph.entity_ids,
      relationIds: result.subgraph.relation_ids,
      stats: {
        entityCount: result.subgraph.entity_ids.length,
        relationCount: expandedCount,
      },
    })
  }

  // Step: LLM Rerank
  if (result.rerank_result) {
    steps.push({
      id: 'rerank',
      label: 'LLM Rerank',
      status: 'completed',
      description: `Selected ${result.rerank_result.selected_relation_ids.length} most relevant relations`,
      entityIds: result.subgraph.entity_ids,
      relationIds: result.rerank_result.selected_relation_ids,
      highlightIds: result.rerank_result.selected_relation_ids,
      stats: {
        entityCount: result.subgraph.entity_ids.length,
        relationCount: result.rerank_result.selected_relation_ids.length,
      },
    })
  }

  // Step: Generate Answer
  steps.push({
    id: 'generate',
    label: 'Generate Answer',
    status: 'completed',
    description: 'LLM generates answer from retrieved context',
    entityIds: result.subgraph.entity_ids,
    relationIds: result.rerank_result?.selected_relation_ids || result.subgraph.relation_ids,
    stats: {
      entityCount: result.subgraph.entity_ids.length,
      relationCount: result.rerank_result?.selected_relation_ids.length || result.subgraph.relation_ids.length,
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
