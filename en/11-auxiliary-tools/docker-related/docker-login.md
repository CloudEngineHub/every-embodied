# Logging In to a Docker Registry

`docker login` connects the local Docker client to a registry so that `docker pull` and `docker push` can access private or restricted images.

## Docker Hub

```bash
docker login
```

Current Docker clients can use device-code authorization with Docker Hub. The terminal prints a verification address and a one-time code. To provide an account name explicitly, run:

```bash
docker login --username <docker-id>
```

## Private Registry

Pass only the registry host and optional port, not an image path:

```bash
docker login registry.example.com
docker login registry.example.com:5000
```

## Token-Based Automation

Use `--password-stdin` so that a token does not appear in shell history or the process list:

```bash
printf '%s' "$REGISTRY_TOKEN" | \
  docker login registry.example.com \
  --username "$REGISTRY_USER" \
  --password-stdin
```

Inject `REGISTRY_TOKEN` through an interactive shell, secret manager, or continuous-integration platform. Do not store it in a tutorial, source repository, or image layer.

## Logout and Verification

```bash
docker info
docker pull hello-world:latest
docker logout
docker logout registry.example.com
```

An `unauthorized` response usually indicates an incorrect registry, account, token scope, or image namespace. See the [Docker CLI login reference](https://docs.docker.com/reference/cli/docker/login/) for registry-specific behavior.
