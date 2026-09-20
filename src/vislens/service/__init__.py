"""The audit service.

Inside the package rather than a top-level `service/` directory so that
`vislens.service.app:app` is importable from any working directory once the
package is installed — which is what lets a launcher, a container, and a test
all name the app the same way.
"""
