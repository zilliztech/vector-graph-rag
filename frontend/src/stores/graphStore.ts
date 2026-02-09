import { create } from 'zustand'
import type { Node, Edge } from '@xyflow/react'
import type {
  SubGraph,
  Entity,
  Relation,
  NodeStatus,
  RetrievalDetail,
  RerankResult,
  StepNodeStatus,
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
  // For handling multiple edges between same node pair
  edgeIndex: number
  totalEdges: number
  // Status of connected nodes (to determine if label should be shown)
  sourceNodeStatus: NodeStatus
  targetNodeStatus: NodeStatus
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

  // Current step status for coloring
  currentNodeStatus: StepNodeStatus | null

  // Selection state
  selectedNodeId: string | null
  hoveredNodeId: string | null

  // Actions
  setSubgraph: (
    subgraph: SubGraph,
    retrievalDetail?: RetrievalDetail,
    rerankResult?: RerankResult
  ) => void
  setStepStatus: (nodeStatus: StepNodeStatus) => void
  setSelectedNode: (nodeId: string | null) => void
  setHoveredNode: (nodeId: string | null) => void
  clearGraph: () => void
}

// Determine entity status based on current step's nodeStatus
function getEntityStatus(
  entityId: string,
  nodeStatus: StepNodeStatus | null
): NodeStatus {
  if (!nodeStatus) return 'undiscovered'

  // Check in order of priority
  if (nodeStatus.seedEntityIds.includes(entityId)) {
    return 'seed'
  }
  if (nodeStatus.expandedEntityIds.includes(entityId)) {
    return 'expanded'
  }

  return 'undiscovered'
}

// Determine relation status based on current step's nodeStatus
function getRelationStatus(
  relationId: string,
  nodeStatus: StepNodeStatus | null
): NodeStatus {
  if (!nodeStatus) return 'undiscovered'

  // Check in order of priority (selected > filtered > expanded > seed > undiscovered)
  if (nodeStatus.selectedRelationIds.includes(relationId)) {
    return 'selected'
  }
  if (nodeStatus.filteredRelationIds.includes(relationId)) {
    return 'filtered'
  }
  if (nodeStatus.seedRelationIds.includes(relationId)) {
    return 'seed'
  }
  if (nodeStatus.expandedRelationIds.includes(relationId)) {
    return 'expanded'
  }

  return 'undiscovered'
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
  entityMap: Map<string, Entity>,
  edgeIndex: number,
  totalEdges: number,
  sourceNodeStatus: NodeStatus,
  targetNodeStatus: NodeStatus
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
      edgeIndex,
      totalEdges,
      sourceNodeStatus,
      targetNodeStatus,
    },
  }
}

// Create a key for edge pairs (order-independent)
function getEdgePairKey(sourceId: string, targetId: string): string {
  return [sourceId, targetId].sort().join('|')
}

// Build nodes and edges with given status
function buildNodesAndEdges(
  subgraph: SubGraph,
  nodeStatus: StepNodeStatus | null
): { nodes: Node<GraphNodeData>[]; edges: Edge<GraphEdgeData>[] } {
  // Start with existing entities
  const entityMap = new Map(subgraph.entities.map((e) => [e.id, e]))

  // Track placeholder entities we need to create
  const placeholderEntities = new Map<string, Entity>()

  // First pass: find all missing entities referenced by relations
  subgraph.relations.forEach((relation) => {
    const [sourceId, targetId] = relation.entity_ids

    // Create placeholder for missing source entity
    if (!entityMap.has(sourceId) && !placeholderEntities.has(sourceId)) {
      placeholderEntities.set(sourceId, {
        id: sourceId,
        name: relation.subject,
        relation_ids: [],
        passage_ids: [],
      })
    }

    // Create placeholder for missing target entity
    if (!entityMap.has(targetId) && !placeholderEntities.has(targetId)) {
      placeholderEntities.set(targetId, {
        id: targetId,
        name: relation.object,
        relation_ids: [],
        passage_ids: [],
      })
    }
  })

  // Merge placeholder entities into entityMap
  placeholderEntities.forEach((entity, id) => {
    entityMap.set(id, entity)
  })

  // Build all nodes with status (existing entities)
  const nodes: Node<GraphNodeData>[] = subgraph.entities.map((entity) => {
    const status = getEntityStatus(entity.id, nodeStatus)
    return entityToNode(entity, status)
  })

  // Add placeholder nodes (always 'undiscovered' status since they're not in the subgraph's entity list)
  placeholderEntities.forEach((entity) => {
    nodes.push(entityToNode(entity, 'undiscovered'))
  })

  // Count edges between each pair of nodes (now all relations should have both endpoints)
  const edgePairCounts = new Map<string, number>()
  subgraph.relations.forEach((relation) => {
    const [sourceId, targetId] = relation.entity_ids
    const key = getEdgePairKey(sourceId, targetId)
    edgePairCounts.set(key, (edgePairCounts.get(key) || 0) + 1)
  })

  // Track current index for each pair
  const edgePairIndices = new Map<string, number>()

  // Build a map of entity status for quick lookup
  const entityStatusMap = new Map<string, NodeStatus>()
  subgraph.entities.forEach((entity) => {
    entityStatusMap.set(entity.id, getEntityStatus(entity.id, nodeStatus))
  })
  // Placeholder entities are always 'undiscovered'
  placeholderEntities.forEach((_, id) => {
    entityStatusMap.set(id, 'undiscovered')
  })

  // Build all edges with status and offset info
  const edges: Edge<GraphEdgeData>[] = []
  subgraph.relations.forEach((relation) => {
    const [sourceId, targetId] = relation.entity_ids

    const key = getEdgePairKey(sourceId, targetId)
    const totalEdges = edgePairCounts.get(key) || 1
    const edgeIndex = edgePairIndices.get(key) || 0
    edgePairIndices.set(key, edgeIndex + 1)

    const status = getRelationStatus(relation.id, nodeStatus)
    const sourceNodeStatus = entityStatusMap.get(sourceId) || 'undiscovered'
    const targetNodeStatus = entityStatusMap.get(targetId) || 'undiscovered'
    const edge = relationToEdge(relation, status, entityMap, edgeIndex, totalEdges, sourceNodeStatus, targetNodeStatus)
    if (edge) {
      edges.push(edge)
    }
  })

  return { nodes, edges }
}

export const useGraphStore = create<GraphState>((set, get) => ({
  // Initial state
  nodes: [],
  edges: [],
  subgraph: null,
  retrievalDetail: null,
  rerankResult: null,
  currentNodeStatus: null,
  selectedNodeId: null,
  hoveredNodeId: null,

  // Actions
  setSubgraph: (subgraph, retrievalDetail, rerankResult) => {
    // Build with all nodes visible (final step status)
    const finalStatus: StepNodeStatus = {
      seedEntityIds: retrievalDetail?.entity_ids || [],
      seedRelationIds: retrievalDetail?.relation_ids || [],
      expandedEntityIds: subgraph.entity_ids.filter(
        id => !retrievalDetail?.entity_ids.includes(id)
      ),
      expandedRelationIds: subgraph.relation_ids.filter(
        id => !retrievalDetail?.relation_ids.includes(id)
      ),
      filteredRelationIds: subgraph.relation_ids.filter(
        id => !rerankResult?.selected_relation_ids.includes(id)
      ),
      selectedRelationIds: rerankResult?.selected_relation_ids || [],
    }

    const { nodes, edges } = buildNodesAndEdges(subgraph, finalStatus)

    set({
      subgraph,
      retrievalDetail: retrievalDetail || null,
      rerankResult: rerankResult || null,
      currentNodeStatus: finalStatus,
      nodes,
      edges,
    })
  },

  setStepStatus: (nodeStatus) => {
    const { subgraph } = get()
    if (!subgraph) return

    const { nodes, edges } = buildNodesAndEdges(subgraph, nodeStatus)

    set({
      currentNodeStatus: nodeStatus,
      nodes,
      edges,
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
      currentNodeStatus: null,
      selectedNodeId: null,
      hoveredNodeId: null,
    }),
}))
