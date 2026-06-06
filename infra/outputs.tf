output "artifact_registry_uri" {
  value = google_artifact_registry_repository.ritam.registry_uri
}

output "cloud_run_service_url" {
  value = google_cloud_run_v2_service.ritam.uri
}