resource "google_artifact_registry_repository" "main" {
  location      = var.region
  repository_id = local.repository_name
  description   = "Repository for ${var.app_name} docker images"
  format        = "DOCKER"

  docker_config {
    immutable_tags = false
  }
}
