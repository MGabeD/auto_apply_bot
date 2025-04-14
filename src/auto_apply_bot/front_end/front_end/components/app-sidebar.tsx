"use client"

import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar"
import { Upload, MessageSquare } from "lucide-react"

interface AppSidebarProps {
  activePage: "upload" | "chat-editor"
  setActivePage: (page: "upload" | "chat-editor") => void
}

export function AppSidebar({ activePage, setActivePage }: AppSidebarProps) {
  return (
    <Sidebar className="fixed left-0 top-0 z-30">
      <SidebarHeader>
        <h1 className="text-lg font-bold px-4 py-2">AI Assistant</h1>
      </SidebarHeader>
      <SidebarContent>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton onClick={() => setActivePage("upload")} isActive={activePage === "upload"}>
              <Upload size={18} />
              <span>Upload Context</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton onClick={() => setActivePage("chat-editor")} isActive={activePage === "chat-editor"}>
              <MessageSquare size={18} />
              <span>Chat & Editor</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarContent>
    </Sidebar>
  )
}
