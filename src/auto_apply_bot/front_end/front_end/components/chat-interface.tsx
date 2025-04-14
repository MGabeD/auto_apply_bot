"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Checkbox } from "@/components/ui/checkbox"
import { MessageSquare, Send } from "lucide-react"
import { sendPrompt } from "@/lib/api"

interface ChatInterfaceProps {
  setResponse: (response: string | null) => void
  setIsLoading: (isLoading: boolean) => void
  includeEditor: boolean
  setIncludeEditor: (include: boolean) => void
  response: string | null
}

export function ChatInterface({
  setResponse,
  setIsLoading,
  includeEditor,
  setIncludeEditor,
  response,
}: ChatInterfaceProps) {
  const [prompt, setPrompt] = useState("")
  const [localResponse, setLocalResponse] = useState<string | null>(null)

  const handleSubmit = async () => {
    if (!prompt.trim()) return

    setIsLoading(true)

    try {
      const response = await sendPrompt(prompt, includeEditor)
      setResponse(response)
      setLocalResponse(response)
    } catch (error) {
      console.error("Error sending prompt:", error)
      setResponse("Error: Failed to get a response from the server.")
      setLocalResponse("Error: Failed to get a response from the server.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageSquare size={20} />
          Chat Interface
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <Textarea
            placeholder="Type your question here..."
            className="min-h-[180px] resize-none"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />

          {/* Only show the checkbox if there's a response */}
          {localResponse !== null && (
            <div className="flex items-center space-x-2">
              <Checkbox
                id="include-editor"
                checked={includeEditor}
                onCheckedChange={(checked) => setIncludeEditor(checked as boolean)}
              />
              <label
                htmlFor="include-editor"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Include editing interface in API request
              </label>
            </div>
          )}

          <Button onClick={handleSubmit} className="w-full" disabled={!prompt.trim()}>
            <Send size={16} className="mr-2" />
            Send Question
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
