output "artifact_registry_uri" {
  value = google_artifact_registry_repository.main.registry_uri
}

output "cloud_run_url" {
  value = google_cloud_run_v2_service.this.uri
}