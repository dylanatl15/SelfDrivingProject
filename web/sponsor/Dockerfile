# Public sponsor page: sponsor.dylantamayo.dev
#
# Build context is the REPOSITORY ROOT, not this directory, because the page is generated
# from `web/sponsor/parts.yaml` and the clips in `media/` at image build time. That is on
# purpose:
# it makes "did you remember to re-run the generator?" impossible to get wrong.
#
#   docker build -f web/sponsor/Dockerfile -t sponsor-page .
#
# From the portfolio stack, see web/sponsor/README.md for the compose service.

FROM python:3.12-alpine AS build
WORKDIR /src
RUN pip install --no-cache-dir "pyyaml>=6"
COPY src/selfdrive/__init__.py src/selfdrive/__init__.py
COPY src/selfdrive/web src/selfdrive/web
COPY web/sponsor/parts.yaml web/sponsor/parts.yaml
# The offer overlay is gitignored and may not exist; the glob always matches the committed
# example, so the build works either way. Without it the page is correct but shows no offer
# badge. Only `dist/` is copied into the serving image below, so no quote ships in it.
COPY web/sponsor/offers.local*.yaml web/sponsor/
COPY media/demo-loop.mp4 media/demo-loop.mp4
COPY media/demo-poster.jpg media/demo-poster.jpg
COPY media/demo.mp4 media/demo.mp4
ENV PYTHONPATH=/src/src
RUN python -m selfdrive.web.sponsor && test -s web/sponsor/dist/index.html

FROM nginx:alpine
COPY --from=build /src/web/sponsor/dist /usr/share/nginx/html
COPY web/sponsor/nginx.conf /etc/nginx/conf.d/default.conf
RUN apk add --no-cache curl
EXPOSE 80
HEALTHCHECK CMD curl --fail http://localhost:80/ || exit 1
CMD ["nginx", "-g", "daemon off;"]
