"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, File, CheckCircle, AlertCircle } from "lucide-react"
import { uploadFile } from "@/lib/api"

export function FileUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<"idle" | "success" | "error">("idle")
  const [uploadMessage, setUploadMessage] = useState("")

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0])
      setUploadStatus("idle")
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    setUploading(true)
    setUploadStatus("idle")

    try {
      await uploadFile(selectedFile)
      setUploadStatus("success")
      setUploadMessage("File uploaded successfully!")
    } catch (error) {
      setUploadStatus("error")
      setUploadMessage("Failed to upload file. Please try again.")
      console.error("Upload error:", error)
    } finally {
      setUploading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload size={20} />
          File Upload
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div
            className="border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:bg-gray-50 transition-colors"
            onClick={() => document.getElementById("file-input")?.click()}
          >
            <input id="file-input" type="file" className="hidden" onChange={handleFileChange} />
            <File className="mx-auto h-12 w-12 text-gray-400" />
            <p className="mt-2 text-sm text-gray-600">Click to select a file or drag and drop</p>
            {selectedFile && <p className="mt-2 text-sm font-medium">Selected: {selectedFile.name}</p>}
          </div>

          <Button onClick={handleUpload} disabled={!selectedFile || uploading} className="w-full">
            {uploading ? "Uploading..." : "Upload to Server"}
          </Button>

          {uploadStatus !== "idle" && (
            <div
              className={`flex items-center gap-2 p-2 rounded text-sm ${
                uploadStatus === "success" ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700"
              }`}
            >
              {uploadStatus === "success" ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
              {uploadMessage}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
