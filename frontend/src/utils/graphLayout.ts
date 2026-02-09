import dagre from '@dagrejs/dagre'
import type { Node, Edge } from '@xyflow/react'

const NODE_WIDTH = 100
const NODE_HEIGHT = 36

export interface LayoutOptions {
  direction?: 'TB' | 'BT' | 'LR' | 'RL'
  nodeSpacing?: number
  rankSpacing?: number
}

export function layoutGraph<T extends Record<string, unknown>, E extends Record<string, unknown>>(
  nodes: Node<T>[],
  edges: Edge<E>[],
  options: LayoutOptions = {}
): Node<T>[] {
  const { direction = 'LR', nodeSpacing = 30, rankSpacing = 80 } = options

  if (nodes.length === 0) return []

  const g = new dagre.graphlib.Graph()
  g.setGraph({
    rankdir: direction,
    nodesep: nodeSpacing,
    ranksep: rankSpacing,
    marginx: 20,
    marginy: 20,
  })
  g.setDefaultEdgeLabel(() => ({}))

  // Add nodes
  nodes.forEach((node) => {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  })

  // Add edges
  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target)
  })

  // Run layout
  dagre.layout(g)

  // Apply positions to nodes
  return nodes.map((node) => {
    const nodeWithPosition = g.node(node.id)
    if (!nodeWithPosition) return node

    return {
      ...node,
      position: {
        x: nodeWithPosition.x - NODE_WIDTH / 2,
        y: nodeWithPosition.y - NODE_HEIGHT / 2,
      },
    }
  })
}

// Force-directed layout for better space utilization
export function forceDirectedLayout<T extends Record<string, unknown>, E extends Record<string, unknown>>(
  nodes: Node<T>[],
  edges: Edge<E>[],
  options: {
    width?: number
    height?: number
    iterations?: number
    repulsion?: number
    attraction?: number
    damping?: number
  } = {}
): Node<T>[] {
  const {
    width = 1200,
    height = 800,
    iterations = 300,
    repulsion = 5000,
    attraction = 0.01,
    damping = 0.9,
  } = options

  if (nodes.length === 0) return []
  if (nodes.length === 1) {
    return [{ ...nodes[0], position: { x: width / 2, y: height / 2 } }]
  }

  // Create adjacency map for edges
  const adjacency = new Map<string, Set<string>>()
  edges.forEach((edge) => {
    if (!adjacency.has(edge.source)) adjacency.set(edge.source, new Set())
    if (!adjacency.has(edge.target)) adjacency.set(edge.target, new Set())
    adjacency.get(edge.source)!.add(edge.target)
    adjacency.get(edge.target)!.add(edge.source)
  })

  // Initialize positions randomly
  const positions = new Map<string, { x: number; y: number }>()
  const velocities = new Map<string, { x: number; y: number }>()

  nodes.forEach((node, index) => {
    // Use golden angle for better initial distribution
    const angle = index * 2.39996323
    const radius = Math.sqrt(index + 1) * 50
    positions.set(node.id, {
      x: width / 2 + radius * Math.cos(angle),
      y: height / 2 + radius * Math.sin(angle),
    })
    velocities.set(node.id, { x: 0, y: 0 })
  })

  // Run force simulation
  for (let iter = 0; iter < iterations; iter++) {
    const temperature = 1 - iter / iterations

    // Calculate repulsive forces between all nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const nodeA = nodes[i]
        const nodeB = nodes[j]
        const posA = positions.get(nodeA.id)!
        const posB = positions.get(nodeB.id)!

        const dx = posB.x - posA.x
        const dy = posB.y - posA.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 1
        const minDist = 80 // Minimum distance between nodes

        // Repulsion force (Coulomb's law)
        const force = (repulsion / (dist * dist)) * temperature
        const fx = (dx / dist) * force
        const fy = (dy / dist) * force

        const velA = velocities.get(nodeA.id)!
        const velB = velocities.get(nodeB.id)!
        velA.x -= fx
        velA.y -= fy
        velB.x += fx
        velB.y += fy

        // Additional repulsion if too close
        if (dist < minDist) {
          const extraForce = ((minDist - dist) / minDist) * 5
          velA.x -= (dx / dist) * extraForce
          velA.y -= (dy / dist) * extraForce
          velB.x += (dx / dist) * extraForce
          velB.y += (dy / dist) * extraForce
        }
      }
    }

    // Calculate attractive forces for connected nodes
    edges.forEach((edge) => {
      const posSource = positions.get(edge.source)
      const posTarget = positions.get(edge.target)
      if (!posSource || !posTarget) return

      const dx = posTarget.x - posSource.x
      const dy = posTarget.y - posSource.y
      const dist = Math.sqrt(dx * dx + dy * dy) || 1

      // Attraction force (Hooke's law)
      const force = dist * attraction * temperature
      const fx = (dx / dist) * force
      const fy = (dy / dist) * force

      const velSource = velocities.get(edge.source)
      const velTarget = velocities.get(edge.target)
      if (velSource && velTarget) {
        velSource.x += fx
        velSource.y += fy
        velTarget.x -= fx
        velTarget.y -= fy
      }
    })

    // Centering force - pull nodes toward center
    nodes.forEach((node) => {
      const pos = positions.get(node.id)!
      const vel = velocities.get(node.id)!
      const dx = width / 2 - pos.x
      const dy = height / 2 - pos.y
      vel.x += dx * 0.001 * temperature
      vel.y += dy * 0.001 * temperature
    })

    // Apply velocities with damping
    nodes.forEach((node) => {
      const pos = positions.get(node.id)!
      const vel = velocities.get(node.id)!

      // Limit velocity
      const speed = Math.sqrt(vel.x * vel.x + vel.y * vel.y)
      const maxSpeed = 50 * temperature
      if (speed > maxSpeed) {
        vel.x = (vel.x / speed) * maxSpeed
        vel.y = (vel.y / speed) * maxSpeed
      }

      pos.x += vel.x
      pos.y += vel.y

      // Apply damping
      vel.x *= damping
      vel.y *= damping

      // Keep within bounds with margin
      const margin = 50
      pos.x = Math.max(margin, Math.min(width - margin, pos.x))
      pos.y = Math.max(margin, Math.min(height - margin, pos.y))
    })
  }

  // Normalize positions to fit in viewport
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  positions.forEach((pos) => {
    minX = Math.min(minX, pos.x)
    maxX = Math.max(maxX, pos.x)
    minY = Math.min(minY, pos.y)
    maxY = Math.max(maxY, pos.y)
  })

  const graphWidth = maxX - minX || 1
  const graphHeight = maxY - minY || 1
  const padding = 100
  const scaleX = (width - padding * 2) / graphWidth
  const scaleY = (height - padding * 2) / graphHeight
  const scale = Math.min(scaleX, scaleY, 1) // Don't scale up

  return nodes.map((node) => {
    const pos = positions.get(node.id)!
    return {
      ...node,
      position: {
        x: padding + (pos.x - minX) * scale,
        y: padding + (pos.y - minY) * scale,
      },
    }
  })
}

// Circular layout for radial arrangements
export function circularLayout<T extends Record<string, unknown>>(
  nodes: Node<T>[],
  centerX: number = 300,
  centerY: number = 300,
  radius: number = 200
): Node<T>[] {
  const count = nodes.length
  if (count === 0) return []
  if (count === 1) {
    return [{ ...nodes[0], position: { x: centerX, y: centerY } }]
  }

  const angleStep = (2 * Math.PI) / count

  return nodes.map((node, index) => {
    const angle = index * angleStep - Math.PI / 2 // Start from top
    return {
      ...node,
      position: {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      },
    }
  })
}

// Grid layout
export function gridLayout<T extends Record<string, unknown>>(
  nodes: Node<T>[],
  columns: number = 4,
  cellWidth: number = 150,
  cellHeight: number = 80,
  startX: number = 50,
  startY: number = 50
): Node<T>[] {
  return nodes.map((node, index) => {
    const row = Math.floor(index / columns)
    const col = index % columns

    return {
      ...node,
      position: {
        x: startX + col * cellWidth,
        y: startY + row * cellHeight,
      },
    }
  })
}
