import { ChatInterface } from "@/components/chat-interface"
import { EditorInterface } from "@/components/editor-interface"

interface ChatEditorPageProps {
  response: string | null
  setResponse: (response: string | null) => void
  isLoading: boolean
  setIsLoading: (isLoading: boolean) => void
  includeEditor: boolean
  setIncludeEditor: (include: boolean) => void
}

export function ChatEditorPage({
  response,
  setResponse,
  isLoading,
  setIsLoading,
  includeEditor,
  setIncludeEditor,
}: ChatEditorPageProps) {
  return (
    <div className="container mx-auto max-w-full p-4 h-[calc(100vh-32px)]">
      <h1 className="text-3xl font-bold mb-6 text-center">Chat & Editor</h1>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 h-[calc(100vh-150px)]">
        <div className="lg:col-span-2 h-full flex flex-col">
          <ChatInterface
            setResponse={setResponse}
            setIsLoading={setIsLoading}
            includeEditor={includeEditor}
            setIncludeEditor={setIncludeEditor}
            response={response}
          />
        </div>
        <div className="lg:col-span-3 h-full flex flex-col">
          <EditorInterface response={response} isLoading={isLoading} />
        </div>
      </div>
    </div>
  )
}
