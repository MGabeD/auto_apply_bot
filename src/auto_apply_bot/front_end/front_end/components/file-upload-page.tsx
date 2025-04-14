import { FileUpload } from "@/components/file-upload"

export function FileUploadPage() {
  return (
    <div className="container mx-auto max-w-3xl p-4 h-[calc(100vh-32px)]">
      <h1 className="text-3xl font-bold mb-8 text-center">Upload Context Files</h1>
      <div className="flex-grow">
        <FileUpload />
      </div>
    </div>
  )
}
