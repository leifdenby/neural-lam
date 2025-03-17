#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = ["graphviz", "loguru"]
# ///

# Standard library
import abc
import argparse
import ast
import os
import sys

# Third-party
import graphviz
from loguru import logger


class ClassHierarchyExtractor(ast.NodeVisitor):
    def __init__(self, base_module, global_class_hierarchy):
        self.class_hierarchy = global_class_hierarchy  # Use global hierarchy to ensure complete detection
        self.class_modules = {}  # Dictionary to store class to module mapping
        self.external_inheritance = (
            []
        )  # List to store external class inheritance
        self.instantiations = []  # List to store class instantiations
        self.references = (
            []
        )  # List to store class references (e.g., via dataclass fields)
        self.memberships = (
            []
        )  # List to store class memberships via type annotations
        self.current_class = None  # Track the current class
        self.imports = (
            {}
        )  # Track imports for resolving external class references
        self.base_module = base_module

    def first_pass(self, tree, file_path):
        module_path = os.path.splitext(file_path)[0].replace(os.sep, ".")
        if module_path.startswith(self.base_module):
            module_path = module_path[len(self.base_module) :]

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self.class_hierarchy[node.name] = []
                self.class_modules[node.name] = module_path

    def second_pass(self, tree):
        self.visit(tree)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports[alias.asname or alias.name] = (
                f"{node.module}.{alias.name}" if node.module else alias.name
            )

    def visit_Import(self, node):
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name

    def visit_ClassDef(self, node):
        self.current_class = node.name
        parents = []

        for base in node.bases:
            if isinstance(base, ast.Name):
                parents.append(base.id)
            elif isinstance(base, ast.Attribute):
                external_name = self.get_full_attribute_name(base)
                self.external_inheritance.append(
                    (self.current_class, external_name)
                )

        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.annotation, (ast.Name, ast.Attribute)):
                    member_class = self.get_class_name_from_node(
                        stmt.annotation
                    )
                    if member_class in self.class_hierarchy:
                        self.memberships.append(
                            (self.current_class, member_class)
                        )
                elif isinstance(stmt.annotation, ast.Subscript):
                    member_class = self.get_class_name_from_node(
                        stmt.annotation.slice
                    )
                    if member_class in self.class_hierarchy:
                        self.memberships.append(
                            (self.current_class, member_class)
                        )

        self.class_hierarchy[self.current_class] = parents
        self.generic_visit(node)
        self.current_class = None

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and self.current_class:
            class_name = self.get_class_name_from_node(node.value.func)
            logger.debug(
                f"Assignment detected in class '{self.current_class}': class_name='{class_name}'"
            )
            resolved_class = self.resolve_imported_class(class_name)
            logger.debug(f"Resolved class: '{resolved_class}'")
            if (
                resolved_class in self.class_hierarchy
                and resolved_class != self.current_class
            ):
                logger.info(
                    f"Detected instantiation: {self.current_class} -> {resolved_class}"
                )
                self.instantiations.append((self.current_class, resolved_class))
        self.generic_visit(node)

    def resolve_imported_class(self, class_name):
        if class_name in self.class_hierarchy:
            return class_name
        for imported_name, full_name in self.imports.items():
            if full_name.split(".")[-1] == class_name:
                return full_name.split(".")[-1]
        return ""

    def get_class_name_from_node(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self.get_full_attribute_name(node).split(".")[-1]
        return ""

    def get_full_attribute_name(self, node):
        if isinstance(node, ast.Attribute):
            return f"{self.get_full_attribute_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Name):
            return node.id
        return ""


def extract_class_hierarchy_from_paths(paths, base_module):
    global_class_hierarchy = {}
    modules = {}
    instantiations = []
    external_inheritance = []
    references = []
    memberships = []
    trees = []
    files = []

    for path in paths:
        if os.path.isfile(path) and path.endswith(".py"):
            files.append(path)
        elif os.path.isdir(path):
            for root, _, file_list in os.walk(path):
                for file in file_list:
                    if file.endswith(".py"):
                        files.append(os.path.join(root, file))

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
            trees.append((file_path, tree))
            extractor = ClassHierarchyExtractor(
                base_module, global_class_hierarchy
            )
            extractor.first_pass(tree, file_path)
            modules.update(extractor.class_modules)

    for file_path, tree in trees:
        extractor = ClassHierarchyExtractor(base_module, global_class_hierarchy)
        extractor.second_pass(tree)
        instantiations.extend(extractor.instantiations)
        external_inheritance.extend(extractor.external_inheritance)
        references.extend(extractor.references)
        memberships.extend(extractor.memberships)

    return (
        global_class_hierarchy,
        modules,
        instantiations,
        external_inheritance,
        references,
        memberships,
    )


def generate_class_diagram(
    class_hierarchy,
    class_modules,
    instantiations,
    external_inheritance,
    references,
    memberships,
    output_file,
):
    dot = graphviz.Digraph(format="png", graph_attr={"rankdir": "BT"})

    for cls, parents in class_hierarchy.items():
        # if the class has any references to abc.ABC, mark it with an asterisk
        is_abc = False
        for ext_inherit in external_inheritance:
            if ext_inherit[0] == cls and ext_inherit[1] == "abc.ABC":
                is_abc = True

        # label = f"{cls}*\n({class_modules.get(cls, '')})"
        label = f"{cls}{'*' if is_abc else ''}\n({class_modules.get(cls, '')})"

        dot.node(cls, label)
        for parent in parents:
            if parent != "ABC":
                dot.edge(cls, parent)

    for ext_inherit in external_inheritance:
        # if the external class is abc.ABC, mark the inheriting class with an
        # asterisk and don't make and edge or node
        print(ext_inherit)
        if ext_inherit[1] == "abc.ABC":
            continue
        dot.node(
            ext_inherit[1],
            ext_inherit[1],
            style="filled",
            fillcolor="lightblue",
            color="blue",
        )
        dot.edge(ext_inherit[0], ext_inherit[1], color="blue")

    for inst in instantiations:
        dot.edge(inst[1], inst[0], style="dotted")

    for mem in memberships:
        dot.edge(mem[1], mem[0], style="dashed", label="member")

    # Add legend
    with dot.subgraph(name="cluster_legend") as legend:
        legend.attr(label="Legend", style="dashed")
        legend.node("inherits", "inherits", shape="plaintext")
        legend.node(
            "inherits_external", "inherits external class", shape="plaintext"
        )
        legend.node("member_of", "member of", shape="plaintext")
        legend.edge("inherits", "inherits", color="black")
        legend.edge("inherits_external", "inherits_external", color="blue")
        legend.edge("member_of", "member_of", style="dotted")

    dot.render(output_file, view=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a class hierarchy diagram from Python files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to Python files or directories to process.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="class_hierarchy",
        help="Output filename for the diagram (default: class_hierarchy).",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is INFO.",
    )
    parser.add_argument(
        "-m",
        "--base-module",
        default="neural_lam.",
        help="Base module to which class module names should be relative.",
    )

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    (
        class_hierarchy,
        class_modules,
        instantiations,
        external_inheritance,
        references,
        memberships,
    ) = extract_class_hierarchy_from_paths(args.paths, args.base_module)
    if not class_hierarchy:
        logger.info("No class definitions found in the specified paths.")
    else:
        generate_class_diagram(
            class_hierarchy,
            class_modules,
            instantiations,
            external_inheritance,
            references,
            memberships,
            args.output,
        )
        logger.info(f"Class hierarchy diagram saved as {args.output}.png")


if __name__ == "__main__":
    main()
