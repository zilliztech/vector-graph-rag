// Simple toast hook
export interface Toast {
  title: string
  description?: string
  variant?: 'default' | 'destructive'
}

export function useToast() {
  return {
    toast: ({ title, description, variant }: Toast) => {
      // For now, just console.log - can be replaced with a proper toast library later
      const message = `${title}${description ? `: ${description}` : ''}`
      if (variant === 'destructive') {
        console.error(message)
        alert(`Error: ${message}`)
      } else {
        console.log(message)
        alert(message)
      }
    },
  }
}
