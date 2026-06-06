resource "google_artifact_registry_repository" "ritam" {
  location      = var.region
  repository_id = "${var.app_name}-repo"
  description   = "Repository for Ritam Docker images"
  format        = "DOCKER"

  docker_config {
    immutable_tags = false

  }
}

resource "google_cloud_run_v2_service" "ritam" {
  name                 = var.app_name
  location             = var.region
  ingress              = "INGRESS_TRAFFIC_ALL"
  invoker_iam_disabled = true
  deletion_protection  = false
  template {
    containers {
      image = "${google_artifact_registry_repository.ritam.registry_uri}/ritam:latest"
      ports {
        container_port = 8000
      }
      env {
        name  = "LLM_API_KEY"
        value = var.llm_api_key
      }
    }
  }
}