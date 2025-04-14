"use client"

import { useState } from "react"
import { AppSidebar } from "@/components/app-sidebar"
import { SidebarProvider } from "@/components/ui/sidebar"
import { FileUploadPage } from "@/components/file-upload-page"
import { ChatEditorPage } from "@/components/chat-editor-page"

export default function Home() {
  const [activePage, setActivePage] = useState<"upload" | "chat-editor">("upload")
  const [response, setResponse] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [includeEditor, setIncludeEditor] = useState(false)

  return (
    <SidebarProvider>
      <div className="flex min-h-screen">
        <AppSidebar activePage={activePage} setActivePage={setActivePage} />
        <div className="flex-1 ml-16 md:ml-64">
          {activePage === "upload" ? (
            <FileUploadPage />
          ) : (
            <ChatEditorPage
              response={response}
              setResponse={setResponse}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
              includeEditor={includeEditor}
              setIncludeEditor={setIncludeEditor}
            />
          )}
        </div>
      </div>
    </SidebarProvider>
  )
}
