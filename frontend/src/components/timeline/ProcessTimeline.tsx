import { useCallback } from 'react'
import { Check, Circle, Loader2 } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/utils/cn'
import { useSearchStore } from '@/stores/searchStore'
import { useGraphStore } from '@/stores/graphStore'
import type { TimelineStep, StepStatus } from '@/types/api'

function ProcessTimeline() {
  const { timelineSteps, currentStepIndex, setCurrentStep, isSearching } = useSearchStore()
  const { setVisibility } = useGraphStore()

  const handleStepClick = useCallback(
    (index: number) => {
      const step = timelineSteps[index]
      if (!step) return

      setCurrentStep(index)
      setVisibility(step.entityIds, step.relationIds, step.highlightIds)
    },
    [timelineSteps, setCurrentStep, setVisibility]
  )

  if (timelineSteps.length === 0 && !isSearching) {
    return (
      <div className="p-4 text-center text-slate-500">
        <p className="text-sm">No search results yet</p>
        <p className="text-xs mt-1">Enter a query to see the retrieval process</p>
      </div>
    )
  }

  return (
    <div className="p-2 space-y-1">
      <div className="px-2 py-1 mb-2">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          Retrieval Process
        </h3>
      </div>

      <AnimatePresence mode="popLayout">
        {timelineSteps.map((step, index) => (
          <TimelineStepItem
            key={step.id}
            step={step}
            index={index}
            isActive={index === currentStepIndex}
            onClick={() => handleStepClick(index)}
          />
        ))}

        {isSearching && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-3 px-3 py-2 text-slate-500"
          >
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">Processing...</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

interface TimelineStepItemProps {
  step: TimelineStep
  index: number
  isActive: boolean
  onClick: () => void
}

function TimelineStepItem({ step, index, isActive, onClick }: TimelineStepItemProps) {
  return (
    <motion.button
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      onClick={onClick}
      className={cn(
        'w-full flex items-start gap-3 px-3 py-2 rounded-md text-left',
        'transition-colors duration-150',
        isActive
          ? 'bg-blue-50 border-l-2 border-blue-500 pl-[10px]'
          : 'hover:bg-slate-50 border-l-2 border-transparent pl-[10px]'
      )}
    >
      {/* Status icon */}
      <div className="flex-shrink-0 mt-0.5">
        <StepStatusIcon status={step.status} />
      </div>

      {/* Step content */}
      <div className="flex-1 min-w-0">
        <div
          className={cn(
            'text-sm font-medium truncate',
            step.status === 'pending' ? 'text-slate-400' : 'text-slate-700'
          )}
        >
          {step.label}
        </div>

        {step.description && (
          <div className="text-xs text-slate-500 mt-0.5 line-clamp-2">
            {step.description}
          </div>
        )}

        {step.stats && step.status !== 'pending' && (
          <div className="flex items-center gap-3 mt-1">
            <span className="text-xs text-slate-400">
              {step.stats.entityCount} entities
            </span>
            <span className="text-xs text-slate-400">
              {step.stats.relationCount} relations
            </span>
          </div>
        )}
      </div>
    </motion.button>
  )
}

function StepStatusIcon({ status }: { status: StepStatus }) {
  const baseClasses = 'w-5 h-5 rounded-full flex items-center justify-center'

  switch (status) {
    case 'completed':
      return (
        <div className={cn(baseClasses, 'bg-emerald-100 text-emerald-600')}>
          <Check className="w-3 h-3" />
        </div>
      )
    case 'active':
      return (
        <div className={cn(baseClasses, 'bg-blue-100 text-blue-600')}>
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            <Circle className="w-3 h-3 fill-current" />
          </motion.div>
        </div>
      )
    case 'pending':
    default:
      return (
        <div className={cn(baseClasses, 'bg-slate-100 text-slate-400')}>
          <Circle className="w-3 h-3" />
        </div>
      )
  }
}

export default ProcessTimeline
