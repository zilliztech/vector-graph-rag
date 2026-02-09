import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface DatasetState {
  // Current selected dataset name
  currentDataset: string | null

  // Actions
  setCurrentDataset: (name: string | null) => void
}

export const useDatasetStore = create<DatasetState>()(
  persist(
    (set) => ({
      currentDataset: null,

      setCurrentDataset: (name) => set({ currentDataset: name }),
    }),
    {
      name: 'vgrag-dataset',
    }
  )
)
