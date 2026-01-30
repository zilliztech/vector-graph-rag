// API Request Types

export interface QueryRequest {
  question: string
  graph_name?: string
  entity_top_k?: number
  relation_top_k?: number
  entity_similarity_threshold?: number
  relation_similarity_threshold?: number
  expansion_degree?: number
}

// API Response Types

export interface QueryResponse {
  question: string
  answer: string
  query_entities: string[]
  subgraph: SubGraph
  retrieved_passages: string[]
  stats: Record<string, unknown>
  retrieval_detail?: RetrievalDetail
  rerank_result?: RerankResult
}

export interface SubGraph {
  entity_ids: string[]
  relation_ids: string[]
  passage_ids: string[]
  entities: Entity[]
  relations: Relation[]
  passages: Passage[]
  expansion_history: ExpansionStep[]
}

export interface Entity {
  id: string
  name: string
  relation_ids: string[]
  passage_ids: string[]
}

export interface Relation {
  id: string
  text: string
  subject: string
  predicate: string
  object: string
  entity_ids: [string, string]
  passage_ids: string[]
}

export interface Passage {
  id: string
  text: string
}

export interface ExpansionStep {
  step: number
  operation: string
  description?: string
  added_entity_ids?: string[]
  added_relation_ids?: string[]
  new_entity_ids?: string[]
  new_relation_ids?: string[]
  total_entities: number
  total_relations: number
}

export interface RetrievalDetail {
  entity_ids: string[]
  entity_texts: string[]
  entity_scores: number[]
  relation_ids: string[]
  relation_texts: string[]
  relation_scores: number[]
}

export interface RerankResult {
  selected_relation_ids: string[]
  selected_relation_texts: string[]
}

// Graph Stats Response
export interface GraphStats {
  entity_count: number
  relation_count: number
  passage_count: number
  graph_name: string
}

// Neighbors Response
export interface NeighborResponse {
  entity_id: string
  neighbors: Entity[]
  relations: Relation[]
}

// Graph Node status for visualization
export type NodeStatus = 'seed' | 'expanded' | 'evicted' | 'selected' | 'default'

// Timeline step status
export type StepStatus = 'pending' | 'active' | 'completed'

export interface TimelineStep {
  id: string
  label: string
  status: StepStatus
  description?: string
  entityIds: string[]
  relationIds: string[]
  highlightIds?: string[]
  stats?: {
    entityCount: number
    relationCount: number
  }
}
