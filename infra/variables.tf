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

variable "app_version" {
  description = "Version of the application"
  type        = string
}