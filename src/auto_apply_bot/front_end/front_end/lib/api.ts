const API_BASE_URL = "http://localhost:8000/api"

// Function to upload a file to the Django server
export async function uploadFile(file: File): Promise<void> {
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch(`${API_BASE_URL}/upload/`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.message || "Failed to upload file")
  }

  return
}

// Function to send a prompt to the Django server
export async function sendPrompt(prompt: string, includeEditor: boolean): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/prompt/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt,
      include_editor: includeEditor,
    }),
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.message || "Failed to get response")
  }

  const data = await response.json()
  return data.response
}

// Function to save an edited response
export async function saveEditedResponse(editedResponse: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/save-edited/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      edited_response: editedResponse,
    }),
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.message || "Failed to save edited response")
  }

  return
}
