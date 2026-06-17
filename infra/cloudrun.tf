# Indexing cloud run service
resource "google_cloud_run_v2_service" "this" {
  name     = var.app_name
  location = var.region

  ingress = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = "${google_artifact_registry_repository.main.registry_uri}/${var.app_name}:${var.app_version}"

      ports {
        container_port = 8000
      }

      dynamic "env" {
        for_each = local.app_secrets
        content {
          name = upper(replace(env.value, "-", "_"))
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.this[env.value].id
              version = "latest"
            }
          }
        }
      }


    }
    service_account = google_service_account.app.email
  }

  depends_on = [google_project_service.run]
}