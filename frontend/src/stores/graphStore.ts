import { create } from 'zustand'
import type { Node, Edge } from '@xyflow/react'
import type {
  SubGraph,
  Entity,
  Relation,
  NodeStatus,
  RetrievalDetail,
  RerankResult,
} from '@/types/api'

// Custom node data type with index signature for React Flow compatibility
export interface GraphNodeData {
  id: string
  label: string
  type: 'entity'
  status: NodeStatus
  relationIds: string[]
  passageIds: string[]
  [key: string]: unknown
}

// Custom edge data type with index signature for React Flow compatibility
export interface GraphEdgeData {
  id: string
  label: string
  text: string
  subject: string
  predicate: string
  object: string
  status: NodeStatus
  passageIds: string[]
  [key: string]: unknown
}

interface GraphState {
  // Graph data
  nodes: Node<GraphNodeData>[]
  edges: Edge<GraphEdgeData>[]

  // Full subgraph data
  subgraph: SubGraph | null
  retrievalDetail: RetrievalDetail | null
  rerankResult: RerankResult | null

  // Visibility filters based on timeline step
  visibleEntityIds: Set<string>
  visibleRelationIds: Set<string>
  highlightIds: Set<string>

  // Selection state
  selectedNodeId: string | null
  hoveredNodeId: string | null

  // Actions
  setSubgraph: (
    subgraph: SubGraph,
    retrievalDetail?: RetrievalDetail,
    rerankResult?: RerankResult
  ) => void
  setVisibility: (
    entityIds: string[],
    relationIds: string[],
    highlightIds?: string[]
  ) => void
  setSelectedNode: (nodeId: string | null) => void
  setHoveredNode: (nodeId: string | null) => void
  clearGraph: () => void
}

// Convert entity to React Flow node
function entityToNode(
  entity: Entity,
  status: NodeStatus
): Node<GraphNodeData> {
  return {
    id: entity.id,
    type: 'entity',
    position: { x: 0, y: 0 }, // Will be set by layout
    data: {
      id: entity.id,
      label: entity.name,
      type: 'entity',
      status,
      relationIds: entity.relation_ids,
      passageIds: entity.passage_ids,
    },
  }
}

// Convert relation to React Flow edge
function relationToEdge(
  relation: Relation,
  status: NodeStatus,
  entityMap: Map<string, Entity>
): Edge<GraphEdgeData> | null {
  const [sourceId, targetId] = relation.entity_ids
  if (!entityMap.has(sourceId) || !entityMap.has(targetId)) {
    return null
  }

  return {
    id: relation.id,
    source: sourceId,
    target: targetId,
    type: 'relation',
    data: {
      id: relation.id,
      label: relation.predicate,
      text: relation.text,
      subject: relation.subject,
      predicate: relation.predicate,
      object: relation.object,
      status,
      passageIds: relation.passage_ids,
    },
  }
}

// Determine node status based on retrieval detail and rerank result
function getNodeStatus(
  id: string,
  retrievalDetail: RetrievalDetail | null,
  rerankResult: RerankResult | null,
  highlightIds: Set<string>
): NodeStatus {
  // Check if highlighted (current step's new additions)
  if (highlightIds.has(id)) {
    return 'selected'
  }

  // Check if it's a seed node
  if (retrievalDetail) {
    if (
      retrievalDetail.entity_ids.includes(id) ||
      retrievalDetail.relation_ids.includes(id)
    ) {
      return 'seed'
    }
  }

  // Check if selected by rerank
  if (rerankResult?.selected_relation_ids.includes(id)) {
    return 'selected'
  }

  return 'expanded'
}

export const useGraphStore = create<GraphState>((set, get) => ({
  // Initial state
  nodes: [],
  edges: [],
  subgraph: null,
  retrievalDetail: null,
  rerankResult: null,
  visibleEntityIds: new Set(),
  visibleRelationIds: new Set(),
  highlightIds: new Set(),
  selectedNodeId: null,
  hoveredNodeId: null,

  // Actions
  setSubgraph: (subgraph, retrievalDetail, rerankResult) => {
    const entityMap = new Map(subgraph.entities.map((e) => [e.id, e]))
    const highlightIds = new Set<string>()

    // Build nodes
    const nodes = subgraph.entities.map((entity) => {
      const status = getNodeStatus(
        entity.id,
        retrievalDetail || null,
        rerankResult || null,
        highlightIds
      )
      return entityToNode(entity, status)
    })

    // Build edges
    const edges: Edge<GraphEdgeData>[] = []
    subgraph.relations.forEach((relation) => {
      const status = getNodeStatus(
        relation.id,
        retrievalDetail || null,
        rerankResult || null,
        highlightIds
      )
      const edge = relationToEdge(relation, status, entityMap)
      if (edge) {
        edges.push(edge)
      }
    })

    set({
      subgraph,
      retrievalDetail: retrievalDetail || null,
      rerankResult: rerankResult || null,
      nodes,
      edges,
      visibleEntityIds: new Set(subgraph.entity_ids),
      visibleRelationIds: new Set(subgraph.relation_ids),
      highlightIds: new Set(),
    })
  },

  setVisibility: (entityIds, relationIds, highlightIdsList) => {
    const { subgraph, retrievalDetail, rerankResult } = get()
    if (!subgraph) return

    const visibleEntityIds = new Set(entityIds)
    const visibleRelationIds = new Set(relationIds)
    const highlightIds = new Set(highlightIdsList || [])

    const entityMap = new Map(subgraph.entities.map((e) => [e.id, e]))

    // Filter and rebuild nodes
    const nodes = subgraph.entities
      .filter((entity) => visibleEntityIds.has(entity.id))
      .map((entity) => {
        const status = highlightIds.has(entity.id)
          ? 'selected'
          : getNodeStatus(entity.id, retrievalDetail, rerankResult, new Set())
        return entityToNode(entity, status)
      })

    // Filter and rebuild edges
    const edges: Edge<GraphEdgeData>[] = []
    subgraph.relations
      .filter((relation) => visibleRelationIds.has(relation.id))
      .forEach((relation) => {
        // Only include edge if both endpoints are visible
        const [sourceId, targetId] = relation.entity_ids
        if (visibleEntityIds.has(sourceId) && visibleEntityIds.has(targetId)) {
          const status = highlightIds.has(relation.id)
            ? 'selected'
            : getNodeStatus(relation.id, retrievalDetail, rerankResult, new Set())
          const edge = relationToEdge(relation, status, entityMap)
          if (edge) {
            edges.push(edge)
          }
        }
      })

    set({
      nodes,
      edges,
      visibleEntityIds,
      visibleRelationIds,
      highlightIds,
    })
  },

  setSelectedNode: (nodeId) => set({ selectedNodeId: nodeId }),

  setHoveredNode: (nodeId) => set({ hoveredNodeId: nodeId }),

  clearGraph: () =>
    set({
      nodes: [],
      edges: [],
      subgraph: null,
      retrievalDetail: null,
      rerankResult: null,
      visibleEntityIds: new Set(),
      visibleRelationIds: new Set(),
      highlightIds: new Set(),
      selectedNodeId: null,
      hoveredNodeId: null,
    }),
}))
