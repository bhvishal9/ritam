variable "llm_api_key" {
  description = "API key for LLM_API_KEY env var"
  type        = string
  sensitive   = true
}

variable "region" {
  description = "GCP region to deploy resources"
  type        = string
  default     = "europe-west3"
}

variable "app_name" {
  description = "Name of the application"
  type        = string
  default     = "ritam"
}