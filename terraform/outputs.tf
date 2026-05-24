output "ecr_url" {
  value = aws_ecr_repository.app_repo.repository_url
}

output "how_to_access" {
  value = "Go to the ECS Console, find the running Task's Public IP, and visit http://[IP]:8501"
}