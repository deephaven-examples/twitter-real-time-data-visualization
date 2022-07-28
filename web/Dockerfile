# Pull the web-plugin-packager image
FROM ghcr.io/deephaven/web-plugin-packager:latest as build


# Specify the plugins you wish to use. You can specify multiple plugins separated by a space, and optionally include the version number, e.g.
# RUN ./pack-plugins.sh <js-plugin-name>[@version] ...
# For a list of published plugins, see https://www.npmjs.com/search?q=keywords%3Adeephaven-js-plugin


# Here is how you would install the matplotlib and table-example plugins
RUN ./pack-plugins.sh @deephaven/js-plugin-matplotlib @deephaven/js-plugin-table-example

# Copy the packaged plugins over
FROM ghcr.io/deephaven/web:${VERSION:-latest}
COPY --from=build js-plugins/ /usr/share/nginx/html/js-plugins/
