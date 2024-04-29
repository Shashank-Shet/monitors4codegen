"""
This file contains the main interface and the public API for multilspy. 
The abstract class LanguageServer provides a factory method, creator that is 
intended for creating instantiations of language specific clients.
The details of Language Specific configuration are not exposed to the user.
"""

import asyncio
import dataclasses
import json
import logging
import os
import pathlib
import threading
from contextlib import asynccontextmanager, contextmanager
from urllib.parse import unquote, urlparse
from .lsp_protocol_handler.lsp_constants import LSPConstants
from .lsp_protocol_handler import lsp_types as LSPTypes

from . import multilspy_types
from .multilspy_logger import MultilspyLogger
from .lsp_protocol_handler.server import (
    LanguageServerHandler,
    ProcessLaunchInfo,
)
from .lsp_protocol_handler import lsp_types
from .multilspy_config import MultilspyConfig, Language
from .multilspy_exceptions import MultilspyException
from .multilspy_utils import PathUtils, FileUtils, TextUtils
from pathlib import PurePath
from typing import Any, AsyncIterator, Iterator, List, Dict, Union, Tuple
from .type_helpers import ensure_all_methods_implemented


@dataclasses.dataclass
class LSPFileBuffer:
    """
    This class is used to store the contents of an open LSP file in memory.
    """

    # uri of the file
    uri: str

    # The contents of the file
    contents: str

    # The version of the file
    version: int

    # The language id of the file
    language_id: str

    # reference count of the file
    ref_count: int

def debug_line(s: str, colnum: int):
    text = f"Line: {s}"
    track_line = "^".rjust(colnum + len("Line: ") + 1)
    return text, track_line

class LanguageServer:
    """
    The LanguageServer class provides a language agnostic interface to the Language Server Protocol.
    It is used to communicate with Language Servers of different programming languages.
    """

    @classmethod
    def create(cls,
               config: MultilspyConfig,
               logger: MultilspyLogger,
               repository_root_path: str,
               **kwargs) -> "LanguageServer":
        """
        Creates a language specific LanguageServer instance based on the given configuration, and appropriate settings for the programming language.

        If language is Java, then ensure that jdk-17.0.6 or higher is installed, `java` is in PATH, and JAVA_HOME is set to the installation directory.

        :param repository_root_path: The root path of the repository.
        :param config: The Multilspy configuration.
        :param logger: The logger to use.

        :return LanguageServer: A language specific LanguageServer instance.
        """
        if config.code_language == Language.PYTHON:
            from monitors4codegen.multilspy.language_servers.jedi_language_server.jedi_server import (
                JediServer,
            )

            return JediServer(config, logger, repository_root_path)
        elif config.code_language == Language.JAVA:
            from monitors4codegen.multilspy.language_servers.eclipse_jdtls.eclipse_jdtls import (
                EclipseJDTLS,
            )

            return EclipseJDTLS(config, logger, repository_root_path)
        elif config.code_language == Language.RUST:
            from monitors4codegen.multilspy.language_servers.rust_analyzer.rust_analyzer import (
                RustAnalyzer,
            )

            return RustAnalyzer(config, logger, repository_root_path)
        elif config.code_language == Language.CSHARP:
            from monitors4codegen.multilspy.language_servers.omnisharp.omnisharp import OmniSharp
            sln_path = kwargs.get('sln_path')
            return OmniSharp(config, logger, repository_root_path, sln_path)
        else:
            logger.error(f"Language {config.code_language} is not supported")
            raise MultilspyException(f"Language {config.code_language} is not supported")

    def __init__(
        self,
        config: MultilspyConfig,
        logger: MultilspyLogger,
        repository_root_path: str,
        process_launch_info: ProcessLaunchInfo,
        language_id: str,
    ):
        """
        Initializes a LanguageServer instance.

        Do not instantiate this class directly. Use `LanguageServer.create` method instead.

        :param config: The Multilspy configuration.
        :param logger: The logger to use.
        :param repository_root_path: The root path of the repository.
        :param cmd: Each language server has a specific command used to start the server.
                    This parameter is the command to launch the language server process.
                    The command must pass appropriate flags to the binary, so that it runs in the stdio mode,
                    as opposed to HTTP, TCP modes supported by some language servers.
        """
        if type(self) == LanguageServer:
            raise MultilspyException(
                "LanguageServer is an abstract class and cannot be instantiated directly. Use LanguageServer.create method instead."
            )

        self.logger = logger
        self.server_started = False
        self.repository_root_path: str = repository_root_path
        self.completions_available = asyncio.Event()

        self.definition_available = asyncio.Event()
        self.references_available = asyncio.Event()
        self.code_actions_available = asyncio.Event()
        self.code_actions_resolutions_available = asyncio.Event()
        self.completions_available = asyncio.Event()
        self.document_diagnostics_available = asyncio.Event()
        self.service_ready_event = asyncio.Event()
        self.initialize_searcher_command_available = asyncio.Event()

        if config.trace_lsp_communication:
            def logging_fn(source, target, msg):
                self.logger.trace(f"LSP: {source} -> {target}: {str(msg)}")

        else:
            def logging_fn(source, target, msg):
                pass

        # cmd is obtained from the child classes, which provide the language specific command to start the language server
        # LanguageServerHandler provides the functionality to start the language server and communicate with it
        self.server: LanguageServerHandler = LanguageServerHandler(process_launch_info, logger=logging_fn)

        self.language_id = language_id
        self.open_file_buffers: Dict[str, LSPFileBuffer] = {}

    @asynccontextmanager
    async def start_server(self) -> AsyncIterator["LanguageServer"]:
        """
        Starts the Language Server and yields the LanguageServer instance.

        Usage:
        ```
        async with lsp.start_server():
            # LanguageServer has been initialized and ready to serve requests
            await lsp.request_definition(...)
            await lsp.request_references(...)
            # Shutdown the LanguageServer on exit from scope
        # LanguageServer has been shutdown
        ```
        """
        self.server_started = True
        yield self
        self.server_started = False

    # TODO: Add support for more LSP features

    @contextmanager
    def open_file(self, relative_file_path: str) -> Iterator[None]:
        """
        Open a file in the Language Server. This is required before making any requests to the Language Server.

        :param relative_file_path: The relative path of the file to open.
        """
        if not self.server_started:
            self.logger.error(
                "open_file called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri in self.open_file_buffers:
            assert self.open_file_buffers[uri].uri == uri
            assert self.open_file_buffers[uri].ref_count >= 1
            self.open_file_buffers[uri].ref_count += 1
            self.logger.debug(f"File {uri} is already open, incrementing ref_count {self.open_file_buffers[uri].ref_count}")
            yield
            self.open_file_buffers[uri].ref_count -= 1
            self.logger.debug(f"File {uri} is already open, decrementing ref_count {self.open_file_buffers[uri].ref_count}")
        else:
            if not pathlib.Path(absolute_file_path).is_file():
                self.logger.warning(f"File {absolute_file_path} does not exist")
                contents = ""
                raise MultilspyException(f"File {absolute_file_path} does not exist")
            else:
                contents = FileUtils.read_file(self.logger, absolute_file_path)

            version = 0
            self.logger.debug(f"Creating buffer with params: {uri} \t {self.language_id}")
            self.open_file_buffers[uri] = LSPFileBuffer(uri, contents, version, self.language_id, 1)

            self.logger.debug(f"Opening file with params: {uri} \t {self.language_id}")
            self.server.notify.did_open_text_document(
                {
                    LSPConstants.TEXT_DOCUMENT: {
                        LSPConstants.URI: uri,
                        LSPConstants.LANGUAGE_ID: self.language_id,
                        LSPConstants.VERSION: version,
                        LSPConstants.TEXT: contents,
                    }
                }
            )
            try:
                yield
            except Exception as e:
                # VVIMP!
                # If an error occurs while a file is opened, it wouldn't be closed without this code.
                self.logger.error("Error in open_file context manager body")
                self.logger.error(str(e))
            self.open_file_buffers[uri].ref_count -= 1

        if self.open_file_buffers[uri].ref_count == 0:
            self.logger.debug(f"Closing file {uri}")
            self.server.notify.did_close_text_document(
                {
                    LSPConstants.TEXT_DOCUMENT: {
                        LSPConstants.URI: uri,
                    }
                }
            )
            del self.open_file_buffers[uri]
            self.logger.debug(f"Deleted file buffer with uri: {uri}")

    # async def create_files(self, relative_file_paths: List[str]) -> None:
    #     """
    #     Create files in the Language Server.

    #     :param relative_file_paths: The relative paths of the files to create.
    #     """
    #     if not self.server_started:
    #         self.logger.error(
    #             "create_files called before Language Server started",
    #         )
    #         raise MultilspyException("Language Server not started")

    #     payload = []
    #     for relative_file_path in relative_file_paths:
    #         absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
    #         uri = pathlib.Path(absolute_file_path).as_uri()

    #         if uri in self.open_file_buffers:
    #             raise MultilspyException(f"File {relative_file_path} is already open")

    #         payload.append({LSPConstants.URI: uri,})

    #     try:
    #         response = await self.server.send.will_create_files(
    #             { 'files': payload }
    #         )
    #         self.server.notify.did_create_files(
    #             { 'files': payload }
    #         )
    #     except MultilspyException as e:
    #         raise MultilspyException(f"Error in create_files: {e}")
    #     return response

    def create_files(self, relative_file_paths: List[str]) -> None:
        """
        Create files in the Language Server.

        :param relative_file_paths: The relative paths of the files to create.
        """
        # TODO: Fix method
        if not self.server_started:
            self.logger.error(
                "create_files called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        payload = []
        for relative_file_path in relative_file_paths:
            absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
            uri = pathlib.Path(absolute_file_path).as_uri()

            if uri in self.open_file_buffers:
                raise MultilspyException(f"File {relative_file_path} is already open")

            payload.append({LSPConstants.URI: uri,})

        try:
            self.server.notify.did_create_files(
                { 'files': payload }
            )
        except MultilspyException as e:
            raise MultilspyException(f"Error in create_files: {e}")

    def save_file(self, relative_file_path: str) -> None:
        """
        Save the file in the Language Server.

        :param relative_file_path: The relative path of the file to save.
        """
        if not self.server_started:
            self.logger.error(
                "save_file called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri not in self.open_file_buffers:
            raise MultilspyException(f"File {relative_file_path} is not open")

        file_buffer = self.open_file_buffers[uri]
        self.server.notify.did_save_text_document(
            {
                LSPConstants.TEXT_DOCUMENT: {
                    LSPConstants.URI: file_buffer.uri,
                }
            }
        )

    def open_file_manual(self, relative_file_path: str):
        """
        Open a file in the Language Server. This is required before making any requests to the Language Server.

        :param relative_file_path: The relative path of the file to open.
        """
        if not self.server_started:
            self.logger.error(
                "open_file called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri in self.open_file_buffers:
            assert self.open_file_buffers[uri].uri == uri
            assert self.open_file_buffers[uri].ref_count >= 1
            self.logger.debug(f"File {uri} is already open, incrementing ref_count {self.open_file_buffers[uri].ref_count}")
            self.open_file_buffers[uri].ref_count += 1
        else:
            if not pathlib.Path(absolute_file_path).is_file():
                self.logger.warning(f"File {absolute_file_path} does not exist")
                contents = ""
                raise MultilspyException(f"File {absolute_file_path} does not exist")
                pass
            else:
                contents = FileUtils.read_file(self.logger, absolute_file_path)
                self.logger.debug(f"File {absolute_file_path} contents:\n {contents}")

            version = 0
            self.logger.debug(f"Creating buffer with params: {uri} \t {self.language_id}")
            self.open_file_buffers[uri] = LSPFileBuffer(uri, contents, version, self.language_id, 1)

            self.logger.debug(f"Opening file with params: {uri} \t {self.language_id}")
            self.server.notify.did_open_text_document(
                {
                    LSPConstants.TEXT_DOCUMENT: {
                        LSPConstants.URI: uri,
                        LSPConstants.LANGUAGE_ID: self.language_id,
                        LSPConstants.VERSION: version,
                        LSPConstants.TEXT: contents,
                    }
                }
            )
        return

    def close_file_manual(self, relative_file_path: str) -> Iterator[None]:
        """
        Open a file in the Language Server. This is required before making any requests to the Language Server.

        :param relative_file_path: The relative path of the file to open.
        """
        if not self.server_started:
            self.logger.error(
                "open_file called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri in self.open_file_buffers:
            assert self.open_file_buffers[uri].uri == uri
            assert self.open_file_buffers[uri].ref_count >= 1
            self.open_file_buffers[uri].ref_count -= 1
            self.logger.debug(f"File {uri} is already open, decrementing ref_count {self.open_file_buffers[uri].ref_count}")
        else:
            self.logger.error(f"Tried closing file {uri} which is not open")
            raise MultilspyException(f"File {relative_file_path} is not open")

        if self.open_file_buffers[uri].ref_count == 0:
            self.logger.debug(f"Closing file: {uri}")
            self.server.notify.did_close_text_document(
                {
                    LSPConstants.TEXT_DOCUMENT: {
                        LSPConstants.URI: uri,
                    }
                }
            )
            del self.open_file_buffers[uri]
            self.logger.debug(f"Deleted file buffer with uri: {uri}")
        return

    def update_open_file(self, relative_file_path: str, new_contents: str) -> None:
        """ Updates the contents of the given opened file as per the Language Server.
        Expects the LSP to be started, and the file to be open. Does not close the file.
        """
        if not self.server_started:
            self.logger.error(
                "open_file called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri not in self.open_file_buffers:
            self.logger.error(f"File {uri} is not open")
            raise MultilspyException(f"File {relative_file_path} is not open")
        self.logger.debug(f"New contents:\n{new_contents}")
        self.open_file_buffers[uri].contents = new_contents
        self.open_file_buffers[uri].version += 1
        file_buffer = self.open_file_buffers[uri]
        line = new_contents.count("\n")
        column = len(new_contents.split("\n")[-1])
        self.logger.debug(f"Change text document with params: {uri} \t {file_buffer.version}")
        self.server.notify.did_change_text_document(
            {
                LSPConstants.TEXT_DOCUMENT: {
                    LSPConstants.VERSION: file_buffer.version,
                    LSPConstants.URI: file_buffer.uri,
                },
                LSPConstants.CONTENT_CHANGES: [
                    {
                        'text': new_contents,
                        # LSPConstants.RANGE: {
                        #     "start": {"line": 0, "character": 0},
                        #     "end": {"line": line, "character": column},
                        # },
                        # LSPConstants.TEXT: new_contents,
                    }
                ],
            }
        )
        return multilspy_types.Position(line=line, character=column)


    def insert_text_at_position(
        self, relative_file_path: str, line: int, column: int, text_to_be_inserted: str
    ) -> multilspy_types.Position:
        """
        Insert text at the given line and column in the given file and return
        the updated cursor position after inserting the text.

        :param relative_file_path: The relative path of the file to open.
        :param line: The line number at which text should be inserted.
        :param column: The column number at which text should be inserted.
        :param text_to_be_inserted: The text to insert.
        """
        if not self.server_started:
            self.logger.error(
                "insert_text_at_position called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        # Ensure the file is open
        assert uri in self.open_file_buffers

        file_buffer = self.open_file_buffers[uri]
        file_buffer.version += 1
        change_index = TextUtils.get_index_from_line_col(file_buffer.contents, line, column)
        file_buffer.contents = (
            file_buffer.contents[:change_index] + text_to_be_inserted + file_buffer.contents[change_index:]
        )
        self.server.notify.did_change_text_document(
            {
                LSPConstants.TEXT_DOCUMENT: {
                    LSPConstants.VERSION: file_buffer.version,
                    LSPConstants.URI: file_buffer.uri,
                },
                LSPConstants.CONTENT_CHANGES: [
                    {
                        LSPConstants.RANGE: {
                            "start": { LSPConstants.LINE: line, LSPConstants.CHARACTER: column},
                            "end": { LSPConstants.LINE: line, LSPConstants.CHARACTER: column},
                        },
                        LSPConstants.TEXT: text_to_be_inserted,
                    }
                ],
            }
        )
        new_l, new_c = TextUtils.get_updated_position_from_line_and_column_and_edit(line, column, text_to_be_inserted)
        return multilspy_types.Position(line=new_l, character=new_c)

    def delete_text_between_positions(
        self,
        relative_file_path: str,
        start: multilspy_types.Position,
        end: multilspy_types.Position,
    ) -> str:
        """
        Delete text between the given start and end positions in the given file and return the deleted text.
        """
        if not self.server_started:
            self.logger.error(
                "insert_text_at_position called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        # Ensure the file is open
        assert uri in self.open_file_buffers

        file_buffer = self.open_file_buffers[uri]
        file_buffer.version += 1
        del_start_idx = TextUtils.get_index_from_line_col(file_buffer.contents, start["line"], start["character"])
        del_end_idx = TextUtils.get_index_from_line_col(file_buffer.contents, end["line"], end["character"])
        deleted_text = file_buffer.contents[del_start_idx:del_end_idx]
        file_buffer.contents = file_buffer.contents[:del_start_idx] + file_buffer.contents[del_end_idx:]
        self.server.notify.did_change_text_document(
            {
                LSPConstants.TEXT_DOCUMENT: {
                    LSPConstants.VERSION: file_buffer.version,
                    LSPConstants.URI: file_buffer.uri,
                },
                LSPConstants.CONTENT_CHANGES: [{LSPConstants.RANGE: {"start": start, "end": end}, "text": ""}],
            }
        )
        return deleted_text

    def get_open_file_text(self, relative_file_path: str) -> str:
        """
        Get the contents of the given opened file as per the Language Server.

        :param relative_file_path: The relative path of the file to open.
        """
        if not self.server_started:
            self.logger.error(
                "get_open_file_text called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        self.logger.debug(f"Retrieving contents for file: {uri}")
        # Ensure the file is open
        if uri not in self.open_file_buffers:
            self.logger.error(f"File {uri} is not open")
            raise MultilspyException(f"File {relative_file_path} is not open")

        file_buffer = self.open_file_buffers[uri]
        self.logger.debug(f"Contents:\n{file_buffer.contents}")
        return file_buffer.contents

    async def go_to_implementation(
        self, relative_file_path: str, line: int, column: int
    ):
        """ Go to the implementation of the symbol at the given line and column in the given file. """
        if not self.server_started:
            self.logger.error(
                "find_function_definition called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri not in self.open_file_buffers:
            self.logger.debug(f"File {uri} is not open")
            raise MultilspyException(f"File {relative_file_path} is not open")

        try:
            self.logger.debug(f"Go to implementation for {uri} at line {line} and column {column}")
            line_text, track_line = debug_line(self.open_file_buffers[uri].contents.split("\n")[line], column)
            self.logger.debug(f"{line_text}")
            self.logger.debug(f"{track_line}")
            response = await self.server.send.implementation(
                {
                    LSPConstants.TEXT_DOCUMENT: { LSPConstants.URI: uri },
                    LSPConstants.POSITION: {LSPConstants.LINE: line, LSPConstants.CHARACTER: column},
                }
            )
            implementation_uri = response["uri"]
            self.logger.debug(f"Implementation URI: {implementation_uri}")
            return implementation_uri, response["range"]
        except Exception as e:
            raise MultilspyException(f"Error in go_to_implementation: {e}")

    async def request_definition(
        self, relative_file_path: str, line: int, column: int,
    ) -> List[multilspy_types.Location]:
        """
        Raise a [textDocument/definition](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_definition) request to the Language Server
        for the symbol at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbol for which definition should be looked up
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return List[multilspy_types.Location]: A list of locations where the symbol is defined
        """

        if not self.server_started:
            self.logger.error(
                "find_function_definition called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri not in self.open_file_buffers:
            self.logger.error(f"File {uri} is not open")
            raise MultilspyException(f"File {relative_file_path} is not open")

        try:
            self.logger.debug(f"Requesting definition for {uri} at line {line} and column {column}")
            line_text, track_line = debug_line(self.open_file_buffers[uri].contents.split("\n")[line], column)
            self.logger.debug(f"{line_text}")
            self.logger.debug(f"{track_line}")
            response = await self.server.send.definition(
                {
                    LSPConstants.TEXT_DOCUMENT: {
                        LSPConstants.URI: uri
                    },
                    LSPConstants.POSITION: {
                        LSPConstants.LINE: line,
                        LSPConstants.CHARACTER: column,
                    },
                }
            )
        except Exception as e:
            raise MultilspyException(f"Error in request_definition: {e}")

        ret: List[multilspy_types.Location] = []
        if isinstance(response, list):
            # response is either of type Location[] or LocationLink[]
            for item in response:
                assert isinstance(item, dict)
                if LSPConstants.URI in item and LSPConstants.RANGE in item:
                    new_item: multilspy_types.Location = {}
                    new_item.update(item)
                    new_item["absolutePath"] = PathUtils.uri_to_path(new_item["uri"])
                    new_item["relativePath"] = str(
                        PurePath(os.path.relpath(new_item["absolutePath"], self.repository_root_path))
                    )
                    ret.append(multilspy_types.Location(new_item))
                elif (
                    LSPConstants.ORIGIN_SELECTION_RANGE in item
                    and LSPConstants.TARGET_URI in item
                    and LSPConstants.TARGET_RANGE in item
                    and LSPConstants.TARGET_SELECTION_RANGE in item
                ):
                    new_item: multilspy_types.Location = {}
                    new_item["uri"] = item[LSPConstants.TARGET_URI]
                    new_item["absolutePath"] = PathUtils.uri_to_path(new_item["uri"])
                    new_item["relativePath"] = str(
                        PurePath(os.path.relpath(new_item["absolutePath"], self.repository_root_path))
                    )
                    new_item["range"] = item[LSPConstants.TARGET_SELECTION_RANGE]
                    ret.append(multilspy_types.Location(**new_item))
                else:
                    assert False, f"Unexpected response from Language Server: {item}"
                self.logger.debug(f"Definition item: {new_item.uri}")
        elif isinstance(response, dict):
            # response is of type Location
            assert LSPConstants.URI in response
            assert LSPConstants.RANGE in response

            new_item: multilspy_types.Location = {}
            new_item.update(response)
            new_item["absolutePath"] = PathUtils.uri_to_path(new_item["uri"])
            new_item["relativePath"] = str(
                PurePath(os.path.relpath(new_item["absolutePath"], self.repository_root_path))
            )
            self.logger.debug(f"Definition item: {new_item.uri}")
            ret.append(multilspy_types.Location(**new_item))

        else:
            assert False, f"Unexpected response from Language Server: {response}"

        return ret

    async def request_references(
        self, relative_file_path: str, line: int, column: int,
    ) -> List[multilspy_types.Location]:
        """
        Raise a [textDocument/references](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_references) request to the Language Server
        to find references to the symbol at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbol for which references should be looked up
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return List[multilspy_types.Location]: A list of locations where the symbol is referenced
        """

        if not self.server_started:
            self.logger.error(
                "find_all_callers_of_function called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri not in self.open_file_buffers:
            raise MultilspyException(f"File {relative_file_path} is not open")

        with self.open_file(relative_file_path):
            try:
                # sending request to the language server and waiting for response
                response = await self.server.send.references(
                    {
                        "context": {"includeDeclaration": False},
                        LSPConstants.TEXT_DOCUMENT: {
                            LSPConstants.URI: pathlib.Path(os.path.join(self.repository_root_path, relative_file_path)).as_uri()
                        },
                        LSPConstants.POSITION: {
                            LSPConstants.LINE: line,
                            LSPConstants.CHARACTER: column},
                    }
                )
            except Exception as e:
                raise MultilspyException(f"Error in request_references: {e}")

        ret: List[multilspy_types.Location] = []
        assert isinstance(response, list)
        for item in response:
            assert isinstance(item, dict)
            assert LSPConstants.URI in item
            assert LSPConstants.RANGE in item

            new_item: multilspy_types.Location = {}
            new_item.update(item)
            new_item["absolutePath"] = PathUtils.uri_to_path(new_item["uri"])
            new_item["relativePath"] = str(
                PurePath(os.path.relpath(new_item["absolutePath"], self.repository_root_path))
            )
            ret.append(multilspy_types.Location(**new_item))

        return ret

    async def resolve_completion(
        self, completion_item: lsp_types.CompletionItem
    ):
        if 'data' not in completion_item or '$$__handler_id__$$' not in completion_item['data']:
            raise MultilspyException("Completion item resolution won't work. See if this field is being removed in request_completoins")
        return await self.server.send.resolve_completion_item(completion_item)

    async def request_completions(
        self, relative_file_path: str, line: int, column: int,
        allow_incomplete: bool = False,
    ) -> List[multilspy_types.CompletionItem]:
        """
        Raise a [textDocument/completion](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_completion) request to the Language Server
        to find completions at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbol for which completions should be looked up
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return List[multilspy_types.CompletionItem]: A list of completions
        """
        if not self.server_started:
            self.logger.error(
                "find_all_callers_of_function called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri not in self.open_file_buffers:
            self.logger.error(f"File {uri} is not open")
            raise MultilspyException(f"File {relative_file_path} is not open")

        try:
            self.logger.debug(f"Requesting completions for {uri} at line {line} and column {column}")
            line_text, track_line = debug_line(self.open_file_buffers[uri].contents.split("\n")[line], column)
            self.logger.debug(f"{line_text}")
            self.logger.debug(f"{track_line}")
            completion_params: LSPTypes.CompletionParams = {
                LSPConstants.POSITION: {
                    LSPConstants.LINE: line,
                    LSPConstants.CHARACTER: column
                },
                LSPConstants.TEXT_DOCUMENT: { LSPConstants.URI: uri },
                "context": {"triggerKind": LSPTypes.CompletionTriggerKind.Invoked},
            }
            response: Union[List[LSPTypes.CompletionItem], LSPTypes.CompletionList, None] = None

            num_retries = 0
            while response is None or (response["isIncomplete"] and num_retries < 30):
                await self.completions_available.wait()
                self.logger.debug(f"Completions request, try: {num_retries}")
                response: Union[
                    List[LSPTypes.CompletionItem], LSPTypes.CompletionList, None
                ] = await self.server.send.completion(completion_params)
                if response is not None and isinstance(response, list):
                    response = {"items": response, "isIncomplete": False}
                num_retries += 1
        except Exception as e:
            raise MultilspyException(f"Error in request_completions: {e}")

        # TODO: Understand how to appropriately handle `isIncomplete`
        if response is None or (response["isIncomplete"] and not(allow_incomplete)):
            self.logger.warning(f"Incompleted completions response when not allowed: {response}")
            return []

        if "items" in response:
            response = response["items"]

        response: List[LSPTypes.CompletionItem] = response

        for item in response:
            assert "insertText" in item or "textEdit" in item
            assert "kind" in item
            if "label" in item:
                item["completionText"] = item["label"]
            elif "insertText" in item:
                item["completionText"] = item["insertText"]
            elif "textEdit" in item and "newText" in item["textEdit"]:
                item["completionText"] = item["textEdit"]["newText"]
            elif "textEdit" in item and "insert" in item["textEdit"]:
                assert False
            else:
                assert False
            pass
        return response


        items = [item for item in response if item["kind"] != LSPTypes.CompletionItemKind.Keyword]
        # TODO: Handle the case when the completion is a keyword

        completions_list: List[multilspy_types.CompletionItem] = []

        for item in items:
            assert "insertText" in item or "textEdit" in item
            assert "kind" in item
            completion_item = {}
            if "detail" in item:
                completion_item["detail"] = item["detail"]

            if "label" in item:
                completion_item["completionText"] = item["label"]
                completion_item["kind"] = item["kind"]
            elif "insertText" in item:
                completion_item["completionText"] = item["insertText"]
                completion_item["kind"] = item["kind"]
            elif "textEdit" in item and "newText" in item["textEdit"]:
                completion_item["completionText"] = item["textEdit"]["newText"]
                completion_item["kind"] = item["kind"]
            elif "textEdit" in item and "range" in item["textEdit"]:
                new_dot_lineno, new_dot_colno = (
                    completion_params["position"]["line"],
                    completion_params["position"]["character"],
                )
                assert all(
                    (
                        item["textEdit"]["range"]["start"]["line"] == new_dot_lineno,
                        item["textEdit"]["range"]["start"]["character"] == new_dot_colno,
                        item["textEdit"]["range"]["start"]["line"] == item["textEdit"]["range"]["end"]["line"],
                        item["textEdit"]["range"]["start"]["character"]
                        == item["textEdit"]["range"]["end"]["character"],
                    )
                )

                completion_item["completionText"] = item["textEdit"]["newText"]
                completion_item["kind"] = item["kind"]
            elif "textEdit" in item and "insert" in item["textEdit"]:
                assert False
            else:
                assert False

            completion_item = multilspy_types.CompletionItem(**completion_item)
            self.logger.debug(f"Completions item: {completion_item['completionText']}")
            completions_list.append(completion_item)

        return [
            json.loads(json_repr)
            for json_repr in set([json.dumps(item, sort_keys=True) for item in completions_list])
        ]

    async def request_document_symbols(
            self,
            relative_file_path: str,
        ) -> Tuple[
            List[multilspy_types.UnifiedSymbolInformation],
            Union[List[multilspy_types.TreeRepr], None]
        ]:
        """
        Raise a [textDocument/documentSymbol](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_documentSymbol) request to the Language Server
        to find symbols in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbols

        :return Tuple[List[multilspy_types.UnifiedSymbolInformation], Union[List[multilspy_types.TreeRepr], None]]: A list of symbols in the file, and the tree representation of the symbols
        """
        if not self.server_started:
            self.logger.error(
                "find_all_callers_of_function called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri not in self.open_file_buffers:
            raise MultilspyException(f"File {relative_file_path} is not open")

        with self.open_file(relative_file_path):
            try:
                response = await self.server.send.document_symbol(
                    {
                        "textDocument": {
                            "uri": pathlib.Path(os.path.join(self.repository_root_path, relative_file_path)).as_uri()
                        }
                    }
                )
            except Exception as e:
                raise MultilspyException(f"Error in request_document_symbols: {e}")

        ret: List[multilspy_types.UnifiedSymbolInformation] = []
        l_tree = None
        assert isinstance(response, list)
        for item in response:
            assert isinstance(item, dict)
            assert LSPConstants.NAME in item
            assert LSPConstants.KIND in item

            if LSPConstants.CHILDREN in item:
                # TODO: l_tree should be a list of TreeRepr. Define the following function to return TreeRepr as well

                def visit_tree_nodes_and_build_tree_repr(tree: LSPTypes.DocumentSymbol) -> List[multilspy_types.UnifiedSymbolInformation]:
                    l: List[multilspy_types.UnifiedSymbolInformation] = []
                    children = tree['children'] if 'children' in tree else []
                    if 'children' in tree:
                        del tree['children']
                    l.append(multilspy_types.UnifiedSymbolInformation(**tree))
                    for child in children:
                        l.extend(visit_tree_nodes_and_build_tree_repr(child))
                    return l

                ret.extend(visit_tree_nodes_and_build_tree_repr(item))
            else:
                ret.append(multilspy_types.UnifiedSymbolInformation(**item))

        return ret, l_tree

    async def request_hover(self,
                            relative_file_path: str,
                            line: int, column: int) -> Union[multilspy_types.Hover, None]:
        """
        Raise a [textDocument/hover](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_hover) request to the Language Server
        to find the hover information at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the hover information
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return None
        """
        if not self.server_started:
            self.logger.error(
                "find_all_callers_of_function called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri not in self.open_file_buffers:
            raise MultilspyException(f"File {relative_file_path} is not open")

        with self.open_file(relative_file_path):
            try:
                response = await self.server.send.hover(
                    {
                        LSPConstants.TEXT_DOCUMENT: {
                            LSPConstants.URI: uri
                        },
                        LSPConstants.POSITION: {
                            LSPConstants.LINE: line,
                            LSPConstants.CHARACTER: column,
                        },
                    }
                )
            except Exception as e:
                raise MultilspyException(f"Error in request_hover: {e}")

        if response is None:
            return None

        assert isinstance(response, dict)

        return multilspy_types.Hover(**response)

    async def request_signature_help(
        self,
        relative_file_path: str,
        line: int, character: int
    ):
        if not self.server_started:
            self.logger.error(
                "find_all_callers_of_function called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri not in self.open_file_buffers:
            self.logger.error(f"File {uri} is not open")
            raise MultilspyException(f"File {relative_file_path} is not open")

        try:
            params: LSPTypes.SignatureHelpParams = {
                LSPConstants.TEXT_DOCUMENT: { LSPConstants.URI: uri },
                LSPConstants.POSITION: {
                    LSPConstants.LINE: line,
                    LSPConstants.CHARACTER: character
                }
            }
            self.logger.debug(f"Requesting signature help for {uri} at line {line} and column {character}")
            line_text, track_line = debug_line(self.open_file_buffers[uri].contents.split("\n")[line], character)
            self.logger.debug(f"{line_text}")
            self.logger.debug(f"{track_line}")
            response = await self.server.send.signature_help(params)
        except Exception as e:
            raise MultilspyException(f"Error in request_signature_help: {e}")
        return response

    async def get_code_actions(
        self,
        relative_file_path: str,
        start: Tuple[int, int],
        end: Tuple[int, int],
        diagnostics: List[LSPTypes.Diagnostic]
    ) -> List[Union[LSPTypes.Command, LSPTypes.CodeAction]]:
        """ Get code actions for the given range and diagnostics in the given file.
        """
        if not self.server_started:
            self.logger.error(
                "find_all_callers_of_function called before Language Server started",
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri not in self.open_file_buffers:
            self.logger.error(f"File {uri} is not open")
            raise MultilspyException(f"File {relative_file_path} is not open")

        self.logger.debug(f"Requesting code actions for {uri} between {start} and {end}")
        for d in diagnostics:
            self.logger.debug(f"Diagnostic item: {d}")
        code_action_params: LSPTypes.CodeActionParams = {
            LSPConstants.TEXT_DOCUMENT: { LSPConstants.URI: uri },
            LSPConstants.RANGE: {
                "start": { LSPConstants.LINE: start[0], LSPConstants.CHARACTER: start[1] },
                "end": { LSPConstants.LINE: end[0], LSPConstants.CHARACTER: end[1] }
            },
            "context": {"diagnostics": diagnostics}
        }
        response = None
        num_retries = 0
        try:
            while response is None:
                await self.code_actions_available.wait()
                self.logger.debug(f"Code actions request, try: {num_retries}")
                response = await self.server.send.code_action(
                    code_action_params
                )
                num_retries += 1
        except Exception as e:
            raise MultilspyException(f"Error in get_code_actions: {e}")
        return response


@ensure_all_methods_implemented(LanguageServer)
class SyncLanguageServer:
    """
    The SyncLanguageServer class provides a language agnostic interface to the Language Server Protocol.
    It is used to communicate with Language Servers of different programming languages.
    """

    def __init__(self, language_server: LanguageServer) -> None:
        self.language_server = language_server
        self.loop = None
        self.loop_thread = None
        self.logger = language_server.logger

    def exists(self, rel_fpath: str) -> bool:
        absolute_file_path = str(PurePath(self.language_server.repository_root_path, rel_fpath))
        return pathlib.Path(absolute_file_path).is_file()

    @classmethod
    def create(
        cls, config: MultilspyConfig, logger: MultilspyLogger, repository_root_path: str, **kwargs
    ) -> "SyncLanguageServer":
        """
        Creates a language specific LanguageServer instance based on the given configuration, and appropriate settings for the programming language.

        If language is Java, then ensure that jdk-17.0.6 or higher is installed, `java` is in PATH, and JAVA_HOME is set to the installation directory.

        :param repository_root_path: The root path of the repository.
        :param config: The Multilspy configuration.
        :param logger: The logger to use.

        :return SyncLanguageServer: A language specific LanguageServer instance.
        """
        return SyncLanguageServer(LanguageServer.create(config, logger, repository_root_path, **kwargs))

    @contextmanager
    def open_file(self, relative_file_path: str) -> Iterator[None]:
        """
        Open a file in the Language Server. This is required before making any requests to the Language Server.

        :param relative_file_path: The relative path of the file to open.
        """
        if not self.exists(relative_file_path):
            # Potentially problematic, test further
            self.create_files([relative_file_path])
        self.logger.debug(f"Opening file: {relative_file_path}")
        with self.language_server.open_file(relative_file_path):
            yield
        self.logger.debug(f"Closing file: {relative_file_path}")


    def open_file_manual(self, relative_file_path: str) -> None:
        if not self.exists(relative_file_path):
            self.create_files([relative_file_path])
        self.language_server.open_file_manual(relative_file_path)
        return

    def close_file_manual(self, relative_file_path: str) -> None:
        self.language_server.close_file_manual(relative_file_path)
        return

    def insert_text_at_position(
        self, relative_file_path: str, line: int, column: int, text_to_be_inserted: str
    ) -> multilspy_types.Position:
        """
        Insert text at the given line and column in the given file and return
        the updated cursor position after inserting the text.

        :param relative_file_path: The relative path of the file to open.
        :param line: The line number at which text should be inserted.
        :param column: The column number at which text should be inserted.
        :param text_to_be_inserted: The text to insert.
        """
        return self.language_server.insert_text_at_position(relative_file_path, line, column, text_to_be_inserted)

    def update_open_file(
        self,
        relative_file_path: str,
        new_contents: str
    ) -> None:
        """
        Update the contents of the given opened file as per the Language Server.

        :param relative_file_path: The relative path of the file to open.
        :param new_contents: The new contents of the file.
        """
        self.logger.debug(f"updating file contents for {relative_file_path}")
        return self.language_server.update_open_file(relative_file_path, new_contents)

    def delete_text_between_positions(
        self,
        relative_file_path: str,
        start: multilspy_types.Position,
        end: multilspy_types.Position,
    ) -> str:
        """
        Delete text between the given start and end positions in the given file and return the deleted text.
        """
        return self.language_server.delete_text_between_positions(relative_file_path, start, end)

    def get_open_file_text(self, relative_file_path: str) -> str:
        """
        Get the contents of the given opened file as per the Language Server.

        :param relative_file_path: The relative path of the file to open.
        """
        self.logger.debug(f"getting file contents for {relative_file_path}")
        return self.language_server.get_open_file_text(relative_file_path)

    @contextmanager
    def start_server(self) -> Iterator["SyncLanguageServer"]:
        """
        Starts the language server process and connects to it.

        :return: None
        """
        self.loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        loop_thread.start()
        ctx = self.language_server.start_server()
        asyncio.run_coroutine_threadsafe(ctx.__aenter__(), loop=self.loop).result()
        yield self
        asyncio.run_coroutine_threadsafe(ctx.__aexit__(None, None, None), loop=self.loop).result()
        self.loop.call_soon_threadsafe(self.loop.stop)
        loop_thread.join()

    def request_definition(self, file_path: str, line: int, column: int) -> List[multilspy_types.Location]:
        """
        Raise a [textDocument/definition](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_definition) request to the Language Server
        for the symbol at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbol for which definition should be looked up
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return List[multilspy_types.Location]: A list of locations where the symbol is defined
        """
        self.logger.debug("request_definition called")
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_definition(file_path, line, column), self.loop
        ).result()
        self.logger.debug("request_definition concluded")
        return result

    def request_references(self, file_path: str, line: int, column: int) -> List[multilspy_types.Location]:
        """
        Raise a [textDocument/references](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_references) request to the Language Server
        to find references to the symbol at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbol for which references should be looked up
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return List[multilspy_types.Location]: A list of locations where the symbol is referenced
        """
        self.logger.debug("request_references called")
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_references(file_path, line, column), self.loop
        ).result()
        self.logger.debug("request_references concluded")
        return result

    def request_completions(
        self, relative_file_path: str, line: int, column: int, allow_incomplete: bool = False
    ) -> List[multilspy_types.CompletionItem]:
        """
        Raise a [textDocument/completion](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_completion) request to the Language Server
        to find completions at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbol for which completions should be looked up
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return List[multilspy_types.CompletionItem]: A list of completions
        """
        self.logger.debug("request_completions called")
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_completions(relative_file_path, line, column, allow_incomplete),
            self.loop,
        ).result()
        self.logger.debug("request_completions concluded")
        return result

    def resolve_completion(
        self, completion_item: lsp_types.CompletionItem
    ):
        self.logger.debug(f"resolve_completion called for {completion_item['label']}")
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.resolve_completion(completion_item),
            self.loop
        ).result()
        self.logger.debug("resolve_completion concluded")
        return result

    def request_document_symbols(self, relative_file_path: str) -> Tuple[List[multilspy_types.UnifiedSymbolInformation], Union[List[multilspy_types.TreeRepr], None]]:
        """
        Raise a [textDocument/documentSymbol](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_documentSymbol) request to the Language Server
        to find symbols in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbols

        :return Tuple[List[multilspy_types.UnifiedSymbolInformation], Union[List[multilspy_types.TreeRepr], None]]: A list of symbols in the file, and the tree representation of the symbols
        """
        self.logger.debug(f"request_document_symbols called for {relative_file_path}")
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_document_symbols(relative_file_path), self.loop
        ).result()
        self.logger.debug(f"request_document_symbols concluded for {relative_file_path}")
        return result

    def request_hover(self, relative_file_path: str, line: int, column: int) -> Union[multilspy_types.Hover, None]:
        """
        Raise a [textDocument/hover](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_hover) request to the Language Server
        to find the hover information at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the hover information
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return None
        """
        self.logger.debug("request_hover called")
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_hover(relative_file_path, line, column), self.loop
        ).result()
        self.logger.debug("request_hover concluded")
        return result

    def create_files(self, rel_fpaths):
        """ Synchoronous wrapper around the create_files method of the LanguageServer class."""
        self.language_server.create_files(rel_fpaths)
        return
        # result = asyncio.run_coroutine_threadsafe(
        #     self.language_server.create_files(rel_fpaths),
        #     self.loop,
        # ).result()
        # return result

    def save_file(self, relative_file_path: str) -> None:
        """
        Save the file in the Language Server.

        :param relative_file_path: The relative path of the file to save.
        """
        self.language_server.save_file(relative_file_path)
        return

    def request_signature_help(
        self,
        relative_fpath: str,
        line: int,
        character: int
    ):
        """ Synchoronous wrapper around the request_signature_help method of the LanguageServer class."""
        self.logger.debug("request_signature_help called")
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_signature_help(relative_fpath, line, character),
            self.loop
        ).result()
        self.logger.debug("request_signature_help concluded")
        return result

    def go_to_implementation(
        self, relative_file_path: str, line: int, column: int, open_file_if_not_open: bool = False
    ):
        """ Synchoronous wrapper around the go_to_implementation method of the LanguageServer class."""
        self.logger.debug("go_to_implementation called")
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.go_to_implementation(relative_file_path, line, column, open_file_if_not_open),
            self.loop
        ).result()
        self.logger.debug("go_to_implementation concluded")
        return result

    def get_code_actions(
        self,
        relative_fpath: str,
        start: Tuple[int, int],
        end: Tuple[int, int],
        diagnostics: List[LSPTypes.Diagnostic]
    ) -> List[Union[LSPTypes.Command, LSPTypes.CodeAction]]:
        """ Get suggestions for code actions to fix the given diagnostics.

        TODO: Add type annotations
        """
        self.logger.debug(f"Getting code actions for {relative_fpath}, beginning")
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.get_code_actions(relative_fpath, start, end, diagnostics),
            self.loop
        ).result()
        self.logger.debug(f"Getting code actions for {relative_fpath}, concluded")
        return result
