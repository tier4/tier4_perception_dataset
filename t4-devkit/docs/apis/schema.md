# `schema`

## Name of schemas

---

<!-- prettier-ignore-start -->
::: t4_devkit.schema.name
    options:
        show_docstring_attributes: true
<!-- prettier-ignore-end -->

## Table of schemas

---

<!-- prettier-ignore-start -->

::: t4_devkit.schema.tables
    options:
        members: ["SchemaBase"]
        show_bases: false

---

::: t4_devkit.schema.tables
    options:
        filters: ["!SchemaBase", "!FileFormat", "!SensorModality", "!SensorChannel", "!VisibilityLevel"]
        merge_init_into_class: false
        show_signature_annotations: false
        show_docstring_attributes: true

### Other items constructing schema table

---

::: t4_devkit.schema.tables
    options:
        members: ["FileFormat", "SensorModality", "SensorChannel", "VisibilityLevel"]
        merge_init_into_class: false
        show_signature_annotations: false
        show_docstring_attributes: true

<!-- prettier-ignore-end -->
