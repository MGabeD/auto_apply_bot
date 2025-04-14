"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Edit, Save, Loader2 } from "lucide-react"
import { saveEditedResponse } from "@/lib/api"

interface EditorInterfaceProps {
  response: string | null
  isLoading: boolean
}

export function EditorInterface({ response, isLoading }: EditorInterfaceProps) {
  const [editedResponse, setEditedResponse] = useState("")
  const [isSaving, setIsSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState<"idle" | "success" | "error">("idle")

  useEffect(() => {
    if (response) {
      setEditedResponse(response)
    }
  }, [response])

  const handleSave = async () => {
    setIsSaving(true)
    setSaveStatus("idle")

    try {
      await saveEditedResponse(editedResponse)
      setSaveStatus("success")
      setTimeout(() => setSaveStatus("idle"), 3000)
    } catch (error) {
      setSaveStatus("error")
      console.error("Error saving edited response:", error)
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Edit size={20} />
          Editing Interface
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-grow flex flex-col">
        <div className="space-y-4 h-full flex flex-col">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-12 flex-grow">
              <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
              <p className="mt-4 text-sm text-gray-500">Waiting for response...</p>
            </div>
          ) : response ? (
            <div className="flex flex-col h-full space-y-4">
              <Textarea
                className="min-h-0 flex-grow"
                value={editedResponse}
                onChange={(e) => setEditedResponse(e.target.value)}
              />
              <Button onClick={handleSave} disabled={isSaving || !editedResponse.trim()} className="w-full">
                {isSaving ? (
                  <>
                    <Loader2 size={16} className="mr-2 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save size={16} className="mr-2" />
                    Save Edited Response
                  </>
                )}
              </Button>

              {saveStatus === "success" && <p className="text-sm text-green-600">Changes saved successfully!</p>}

              {saveStatus === "error" && (
                <p className="text-sm text-red-600">Failed to save changes. Please try again.</p>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center flex-grow">
              <Edit className="h-8 w-8 text-gray-400" />
              <p className="mt-4 text-sm text-gray-500">
                No response to edit yet. Ask a question in the chat interface.
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
