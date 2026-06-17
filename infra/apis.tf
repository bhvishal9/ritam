# Enable Cloud Run API
resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

# Enable Secrets Manager API
resource "google_project_service" "secrets_manager" {
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}