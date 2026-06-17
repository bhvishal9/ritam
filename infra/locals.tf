locals {
  repository_name = "${var.app_name}-docker-repo"

  app_secrets = [
    "llm_api_key",
    "qdrant_api_key",
    "qdrant_url",
  ]
}
