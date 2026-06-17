resource "google_secret_manager_secret" "this" {
  for_each = toset(local.app_secrets)

  secret_id = each.value
  replication {
    auto {}
  }
}