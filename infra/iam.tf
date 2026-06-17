# # Create a dedicated service account
resource "google_service_account" "app" {
  account_id   = "${var.app_name}-sa"
  display_name = "${var.app_name} Service Account"
}

resource "google_secret_manager_secret_iam_member" "this" {
  for_each = toset(local.app_secrets)

  secret_id  = google_secret_manager_secret.this[each.value].id
  role       = "roles/secretmanager.secretAccessor"
  member     = "serviceAccount:${google_service_account.app.email}"
  depends_on = [google_secret_manager_secret.this]
}